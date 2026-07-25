import torch
import pandas as pd
import numpy as np
import sys
import joblib
from collections import defaultdict
import gc
import tempfile
from Bio import SeqIO
from scipy import stats
import subprocess
from esm.data import FastaBatchedDataset
import os
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 

'''
Script containing functions relevant to attention analysis

NOTE: IUPred2a is required: please install and update filepath directly below directed to Iupred2a

'''

iupred_filepath = '/rds/user/ln399/hpc-work/llps/DeePhase_LN/__PREDICT_LN/tools/iupred2a.py'


def read_fasta(fasta_filepath):
    # Create dictionary: UniProt ID -> Sequence
    seq_dict = {record.id: str(record.seq) for record in SeqIO.parse(f'{fasta_filepath}.fasta', "fasta")}
    
    # Convert to DataFrame: split UniProt ID to keep things clean
    df = pd.DataFrame([
        {"UniProt_ID": rec_id.split("|")[1] if "|" in rec_id else rec_id, "Sequence": seq}
        for rec_id, seq in seq_dict.items()
    ])
    return df

def extract_IDR(seq):

    '''
    This function is from https://github.com/kadiliissaar
    Please update the filepath to your own copy iupred2a
    '''
    tmp_IDR = tempfile.NamedTemporaryFile()  
    with open(tmp_IDR.name, 'w') as f_IDR:
         f_IDR.write('>1\n' + str(seq))
    tmp_IDR.seek(0)
    
    out = subprocess.Popen(['python', f'{iupred_filepath}', str(tmp_IDR.name), 'long'], 
           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_IDR, stderr_IDR = out.communicate()
    stdout_IDR = stdout_IDR.split()[40:]
    
    IDR_prob = []
    for i in range(0, int(len(stdout_IDR)/3)):
        IDR_prob.append(float(str(stdout_IDR[3*i + 2], 'utf-8')))
       
    TH1 = 0.5
    TH2 = 20
    AAs   = pd.Series(list(map(lambda i:i, seq)))    
    IDR_residues = []
    current = 0
    for t in range(0, len(IDR_prob)):
        if IDR_prob[t] > TH1:
            current = current + 1
            if t == len(IDR_prob) - 1:
                if current > TH2:
                    IDR_residues.extend(range(max(t - current, 0), t + 1))
        else:
            if current > TH2:
                IDR_residues.extend(range(t - current , t))
                current = 0
            else:
                current = 0
    
    return len(IDR_residues)

def extract_IDR_profile(seq):

    '''
    This function is from https://github.com/kadiliissaar
    '''

    tmp_IDR = tempfile.NamedTemporaryFile()  
    with open(tmp_IDR.name, 'w') as f_IDR:
         f_IDR.write('>1\n' + str(seq))
    tmp_IDR.seek(0)
    
    out = subprocess.Popen(['python', f'{iupred_filepath}', str(tmp_IDR.name), 'long'], 
           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout_IDR, stderr_IDR = out.communicate()
    stdout_IDR = stdout_IDR.split()[40:]
    
    IDR_prob = []
    for i in range(0, int(len(stdout_IDR)/3)):
        IDR_prob.append(float(str(stdout_IDR[3*i + 2], 'utf-8')))
    
    return IDR_prob
    

def get_attention_distribution_df(protein_dict, normalize = False, standardize = False):
    '''
    Function to get the attention rollout received distribution of each amino acid across all proteins in the rollout matrix 

    Args:  
        protein_dict(dict): Dictionary containing protein information, where each key is a protein ID and the value is a dictionary with keys 'rollout' (attention matrix) and 'sequence' (protein sequence).
        normalize(bool, optional): Whether to normalize the attention values. If True, the attention values will be min-max nornalized. Defaults to False.
        standardize(bool, optional): Whether to standardize the attention values. If True, the attention values will be standardized (mean=0, std=1). Defaults to False. 
    
    Returns: 
        pd.DataFrame with columns ['protein_id', 'amino_acid', 'attention']
    '''

    records = []

    for protein in protein_dict.keys():
        attn_matrix = protein_dict[protein]['rollout']
        seq = protein_dict[protein]['sequence']
        labels = [f"{aa}_{i}" for i, aa in enumerate(seq.strip().upper(), 1)]
        
        attn_matrix = attn_matrix.detach().cpu().numpy() # convert matrix to numpy
        attn_vector = attn_matrix.sum(axis=0) # sum each column 

        if standardize == True:
            mu = attn_vector.mean()
            sigma = attn_vector.std()
            if sigma > 0:
                attn_vector = (attn_vector - mu) / sigma
            else:
                attn_vector = attn_vector - mu  # fallback
        elif normalize == True: 
            attn_vector = (attn_vector - attn_vector.min()) / (attn_vector.max() - attn_vector.min())

        for label, attn in zip(labels, attn_vector):
            aa = label.split('_')[0]
            records.append({
                'protein_id': protein,
                'amino_acid': aa,
                'attention': attn
            })
    return pd.DataFrame(records)

def attention_rollout(attentions, raw_sequence_lengths, discard_ratio=0.9, head_fusion="mean", include_cls = False, device = 'cpu'):
    """
    Compute attention rollout excluding special tokens (<cls>, <eos>, padding).

    Args:
        attentions(tensor): tensor of attention matrices (shape should be 33, 20, x,x) where x is padded sequence length 
        raw_sequence_lengths: length of the protein without any special tokens 
        discard_ratio(float, optional): Fraction of lowest attention values to discard. Defaults to 0.9. 
        head_fusion(chr, optional): How to combine multiple heads ("mean", "max", "min"). Defaults to 'mean'. 
        include_cls(bool, optional): Whether to include the <cls> token in the attention rollout. Defaults to False.
        device(chr, optional): device to use. Defaults to 'cpu'

    Returns:
        Torch.tensor: Final attention rollout tensor of shape [actual_seq_len, actual_seq_len]
    """
    # combine heads in each layer 
    # attentions is of shape (33, 20, seq_len, seq_len) where 33 is the number of layers and 20 is the number of heads
    joint_attentions = []
    for attention in attentions: # for each layer of the attention matrix; attention is of shape (20, seq_len, seq_len)
       
        # the resulting joint_attention below is of shape (seq_len, seq_len)
        # you must use dim = 0 here because you are combining the 20 heads in each layer
        # if you did dim = 1, you would get a resulting joint_attention of shape (20, seq_len)
        # if you did dim = 2, you would get a resulting joint_attention of shape (20, seq_len)

        if head_fusion == "mean":
            joint_attention = attention.mean(dim=0)
        elif head_fusion == "max":
            joint_attention = attention.max(dim=0)[0]
        elif head_fusion == "min":
            joint_attention = attention.min(dim=0)[0]
        else:
            raise ValueError("head_fusion must be 'mean', 'max', or 'min'")
        joint_attentions.append(joint_attention)

        # joint_attentions is a list of tensors, each of shape (seq_len, seq_len), where each tensor corresponds to a layer's head-fused attention matrix
    
    # discard ratio 
    # for each layer's head-averaged attention
    joint_attentions_clean = []
    for layer_attn in joint_attentions:
        seq_len = layer_attn.shape[0]

        actual_len = raw_sequence_lengths
        start_idx = 1  # Skip <cls>

        if include_cls == True: 
            start_idx = 0  # Include <cls> token
            actual_len += 1  # Adjust length to include <cls>

        end_idx = start_idx + actual_len

        # Zero out rows/columns outside actual protein region
        mask = torch.zeros_like(layer_attn) # get a matrix of zeros with the same shape as layer_attn
        mask[start_idx:end_idx, start_idx:end_idx] = 1 # set the region corresponding to the actual protein sequence to 1
        layer_attn *= mask # this *= operator just means (the expression on the left == itself times the expression on the right )

        # Apply discard ratio only on valid protein region
        protein_attn = layer_attn[start_idx:end_idx, start_idx:end_idx]
        flat_att = protein_attn.flatten()
        if flat_att.numel() > 0:
            threshold = torch.quantile(flat_att, discard_ratio) # get the 90th quantile value 
            protein_attn *= (protein_attn >= threshold) # the left here is just the raw attn matrix of actual_seq_len x actual_seq_len and the right is a mask of bools corresponding to if the attn at that position is > than the threshold
            layer_attn[start_idx:end_idx, start_idx:end_idx] = protein_attn

        I = torch.eye(actual_len).to(device)  # create identity matrix
        attn_with_residual = (protein_attn + I) / 2 # add identity matrix to represent residual connections
        attn_normalized = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        joint_attentions_clean.append(attn_normalized)

    rollout = joint_attentions_clean[0]
    for joint_attention in joint_attentions_clean[1:]:
        rollout = torch.matmul(joint_attention, rollout)

    return rollout

def get_attn_rollout(seq, tokenizer, model, device = 'cpu',): 
    '''
    Function to get attention rollout received.

    Args: 
        seq(chr): protein sequence 
        tokenizer(optional): Tokenizer to encode the sequence.
        model(optional): Model used to get attention matrices. 
        device(chr, optional): Which device to use. Defaults to 'cpu'. 

    Returns: 
        numpy vector of attention rollout received of length sequence length. 

    '''
    tokens = tokenizer.encode(seq)
    input_ids = torch.tensor([tokens])
    input_ids = input_ids.to(device)


    with torch.no_grad():
        outputs = model(input_ids = input_ids, output_attentions = True)
        
    # Extract attention weights - list of [batch, num_heads, seq_len, seq_len] tensors
    attention_weights = outputs.attentions

    attn_tensor = torch.stack(attention_weights)[:,0,:,:,:]
    rollout = attention_rollout(attn_tensor, len(seq), discard_ratio=0.9, head_fusion="mean", device = device)
    attention_roll_received = rollout.sum(dim=0)
    attention_roll_received = attention_roll_received.cpu().numpy()
    return attention_roll_received


def safe_load(path):
    gc.collect()
    return joblib.load(path)

def get_head_layer_attn(protein, attn, layer_idx, head_idx): 
    '''

    Function to get the attention matrix for 1 protein at a specific head j and layer i
    This cleans the attention matrix by removing cls and eos tokens from the resulting 
    matrix.

    Args: 
        protein(str) : protein id for which to get the 20 attention heads for each layer 
        attn(dict): dictionary containing attn matrix for the desired protein. The 'protein' variable should be a key in the attn dict 
        layer_idx(int) : which layer you want to get the attn matrix for 
        head_idx(int) : which head you want to get the attn matrix for 
    Returns: 
        attn_matrix_clean(tensor): attention matrix of size (seq_len, seq_len)

    '''

    attn_matrix = attn[protein]['attention_matrix'][layer_idx, head_idx]
    actual_seq_len = attn[protein]['sequence_length']
    attn_matrix_clean = attn_matrix[1:actual_seq_len+1, 1:actual_seq_len+1]

    return attn_matrix_clean
    

def get_aa_attn_perc(attn_matrix, seq, normalize = 'local'):
    '''
    Function to look at attention received at a specific head/layer by each amino acid. 

    Args: 
        attn_matrix(tensor): attention matrix of size (seq_len, seq_len). Please only include the attention matrix after cls and eos tokens have been removed!
        seq(str): protein sequence of the protein that the attention matrix corresponds to
        normalize (str): Method of normalization. Options include ['local', 'occurrence', 'global']. 
            If normalize == 'local', it will calculate the percentage of the total attention budget for the protein received by each AA. 
            If normalize == 'occurrence', it returns the sum of attn received by each AA divided by the total # of occurrences of that AA in the proteins seq/ 
            If normalize == 'global', it just returns the sum of attn received by each AA across the residues in the protein sequence. 

    Returns: 
        df(pd.DataFrame) with columns 'aa' and one of either ['attn_percent', 
        'attn_occur_norm', 'sum_attn'] depending on the normalization strategy. 
    '''    
    attn_vector = attn_matrix.sum(axis=0) # sum each column to get attention received by each AA

    # sum the attention received by each type of amino acid over the sequence 
    aa_attn = defaultdict(float)
    for j, aa in enumerate(seq):
        aa_attn[aa] += attn_vector[j]

    # Normalize
    if normalize == 'local':
        total_attention = attn_vector.sum()
        aa_percentage = {aa: (att / total_attention) * 100 for aa, att in aa_attn.items()}

        df = pd.DataFrame({
            'aa': list(aa_percentage.keys()),
            'attn_percent': list(aa_percentage.values())
        })

    elif normalize == 'occurrence': 
        # Normalize by the number of occurrences of each amino acid in the sequence
        aa_counts = {aa: seq.count(aa) for aa in aa_attn.keys()}
        for aa in aa_attn.keys():
            if aa_counts[aa] > 0:
                aa_attn[aa] /= aa_counts[aa]

        df = pd.DataFrame({
            'aa': list(aa_attn.keys()),
            'attn_occur_norm': list(aa_attn.values())
        })
    elif normalize == 'global': 
        df = pd.DataFrame({
            'aa': list(aa_attn.keys()),
            'sum_attn': list(aa_attn.values())
        })

    else: 
        print('normalize must be "local", "global" or "occurrence"')
        sys.exit()
    return df 


def run_aa_analysis_head_layer(normalize = 'local', model = 'original'): 
    '''
    Function to analyze the attention received by each AA at each head and layer 

    Args: 
        normalize(str, optional): how to normalize the attention values. Options include local, global, and occurrence.  normalize = 'local' will return 
        the percentage of attention received by each amino acid in each sequence at each head/layer. normalize = 'global' will return attn received by each AA at head_i, layer_j divided by the 
        total attention in the entire matrix (across all heads and layers). normalize = 'occurrence' will return the sum of attn received by each AA in one protein seq divided by the number of occurrenes of the AA in that protein seq. 
        Defaults to 'local'.
        model(str, optional): whether to use original or finetuned model attention. Defaults to original. 

    '''
    aa_df = []
    for chunk in range(0,22): # load each chunk of attention matrices
        attn_i = safe_load(f'{BASE_PATH}/attention_matrices_chunks/attention_matrices_labeled_{chunk}_{model}.joblib')
        
        # for each protein in the chunk get the AA attn percentage at each head/layer
        for protein in attn_i.keys(): # for each protein in the batch
            seq_len = attn_i[protein]['sequence_length']
            seq = attn_i[protein]['sequence']
            if normalize == 'global':
                total_attn_glob = attn_i[protein]['attention_matrix'][:, :, 1:seq_len+1, 1:seq_len+1].flatten().sum() # get full attn matrix (across heads and layers) removing cls and eos


            for layer_idx in range(0,33): # for each layer 
                for head_idx in range(0,20): # for each head 

                    head_layer_attn = get_head_layer_attn(protein, attn_i, layer_idx, head_idx)
                    
                    if normalize == 'local': 
                        aa_attn_df = get_aa_attn_perc(head_layer_attn, seq, normalize = 'local')

                    elif normalize == 'global': 
                        aa_attn_df = get_aa_attn_perc(head_layer_attn, seq, normalize = 'global')
                        aa_attn_df['attn_percent'] = (aa_attn_df['sum_attn'] / total_attn_glob) * 100
                        aa_attn_df = aa_attn_df.drop(columns=['sum_attn'])

                    elif normalize == 'occurrence':
                        aa_attn_df = get_aa_attn_perc(head_layer_attn, seq, normalize = 'occurrence')

                    else: 
                        print('normalize must be  "local", "global or "occurrence"')
                        sys.exit()

                    aa_attn_df['protein'] = protein
                    aa_attn_df['layer'] = layer_idx
                    aa_attn_df['head'] = head_idx
                    aa_df.append(aa_attn_df)

                    del head_layer_attn, aa_attn_df

        del attn_i

    return pd.concat(aa_df, ignore_index=True)

def get_esm2_embeddings(df, model, alphabet, device, sequence_col = "sequence", repr_layer = 33, seq_length =  1022, tokens_per_batch = 4096) :

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        fasta_path = f.name
        for idx, seq in enumerate(df[sequence_col].tolist()):
            f.write(f">{idx}\n{seq}\n")

    try:
        dataset = FastaBatchedDataset.from_file(fasta_path)
        batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(seq_length),
            batch_sampler=batches
        )

        embeddings_by_idx = {}

        with torch.no_grad():
            for labels, strs, toks in data_loader:
                toks = toks.to(device)
                out = model(toks, repr_layers=[repr_layer], return_contacts=False)
                representations = out["representations"][repr_layer].to("cpu")

                for i, label in enumerate(labels):
                    idx = int(label.split()[0])                    
                    truncate_len = min(seq_length, len(strs[i]))   
                    mean_emb = representations[i, 1:truncate_len + 1].mean(0).numpy()
                    embeddings_by_idx[idx] = mean_emb

    finally:
        os.unlink(fasta_path)

    # Re-align embeddings with original df row order
    ordered_embeddings = [embeddings_by_idx[i] for i in range(len(df))]
    emb_df = pd.DataFrame(ordered_embeddings)

    return pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    
def fast_statistical_comparison(attn_df):
    # Index for fast lookups
    attn_indexed = attn_df.set_index(['aa', 'layer', 'head', 'Group']).sort_index()
    
    results = []
    combinations = [(aa, layer, head) 
                   for aa in attn_df['aa'].unique()
                   for layer in range(33)
                   for head in range(20)]
    
    for aa, layer, head in combinations:
        try:
            disprot_data = attn_indexed.loc[(aa, layer, head, 'DisProt'), 'attn_occur_norm'].values
            pdb_data = attn_indexed.loc[(aa, layer, head, 'PDB'), 'attn_occur_norm'].values
            
            if len(disprot_data) < 3 or len(pdb_data) < 3:
                continue
            
            # Compute statistics
            disprot_mean = np.mean(disprot_data)
            disprot_std = np.std(disprot_data, ddof=1)  # Sample std (n-1)
            pdb_mean = np.mean(pdb_data)
            pdb_std = np.std(pdb_data, ddof=1)  # Sample std (n-1)
            difference = disprot_mean - pdb_mean
            
            # Cohen's d with proper pooled standard deviation
            n_disprot = len(disprot_data)
            n_pdb = len(pdb_data)
            pooled_std = np.sqrt(
                ((n_disprot - 1) * disprot_std**2 + (n_pdb - 1) * pdb_std**2) / 
                (n_disprot + n_pdb - 2)
            )
            cohens_d = difference / pooled_std if pooled_std > 0 else 0
            
            # Use t-test (much faster than Mann-Whitney)
            statistic, p_value = stats.ttest_ind(disprot_data, pdb_data, equal_var = False)
            
            results.append({
                'amino_acid': aa,
                'layer': layer,
                'head': head,
                'disprot_mean': disprot_mean,
                'disprot_std': disprot_std,
                'pdb_mean': pdb_mean,
                'pdb_std': pdb_std,
                'difference': difference,
                'cohens_d': cohens_d,
                'p_value_uncorrected': p_value,
                'n_disprot': n_disprot,
                'n_pdb': n_pdb
            })
        except KeyError:
            continue
    
    return pd.DataFrame(results)


def cohens_d(distr, g1 = 'DisProt', g2 = 'PDB'): 
    mean_df = distr.groupby(['Group', 'amino_acid'])['attention'].mean().reset_index().rename(columns = {'attention' : 'mean_attention'})
    mean_df = mean_df.pivot(columns = 'Group', values = 'mean_attention', index = 'amino_acid').reset_index()
    mean_df['difference'] = mean_df[g1] - mean_df[g2]
   
    std_df = distr.groupby(['Group', 'amino_acid'])['attention'].std(ddof=1).reset_index().rename(columns = {'attention' : 'std_attention'})
     
    results = []
    for aa in mean_df.amino_acid.unique(): 
        
        n_g1 = len(distr[(distr['Group'] == g1) & (distr['amino_acid'] == aa)])
        n_g2 = len(distr[(distr['Group'] == g2) & (distr['amino_acid'] == aa)])

        mean_df_temp = mean_df[mean_df['amino_acid'] == aa]
        mean_diff = mean_df_temp['difference'].iloc[0]

        g1_std = std_df[(std_df['Group'] == g1) & ( std_df['amino_acid'] == aa)]['std_attention'].iloc[0]
        g2_std = std_df[(std_df['Group'] == g2) & ( std_df['amino_acid'] == aa)]['std_attention'].iloc[0]

        pooled_std = np.sqrt(
                ((n_g1 - 1) * (g1_std**2) + (n_g2 - 1) * (g2_std**2)) / 
                (n_g1 + n_g2 - 2)
            )


        d = mean_diff / pooled_std if pooled_std > 0 else 0

        statistic, p_value = stats.ttest_ind(distr[(distr['Group'] == g1) & (distr['amino_acid'] == aa)]['attention'], distr[(distr['Group'] == g2) & (distr['amino_acid'] == aa)]['attention'], equal_var = False)

        results.append({
                'amino_acid': aa,
                'g1_mean': mean_df_temp[g1].iloc[0],
                'g1_std': g1_std,
                'g2_mean': mean_df_temp[g2].iloc[0],
                'g2_std': g2_std,
                'difference': mean_diff,
                'cohens_d': d,
                'p_value_uncorrected': p_value,
                'n_g1': n_g1,
                'n_g2': n_g2
            })

    return pd.DataFrame(results)