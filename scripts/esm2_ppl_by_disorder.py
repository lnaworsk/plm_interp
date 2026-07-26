import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import math
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
import statsmodels.api as sm
from attention_functions import extract_IDR_profile
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''

Script for computing linear regression of perplexity ~ disorder 

'''

# ------------------------------------
#### load ESM2 from Hugging Face ####
# ------------------------------------

model_name = "facebook/esm2_t33_650M_UR50D"

print("Loading model and tokenizer...")
model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only = True, trust_remote_code = False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = False ,local_files_only = True )
model = model.to(device)


# ------------------------------------
#### load proteins and tokenize ####
# ------------------------------------

proteins_df = pd.read_csv(f'{BASE_PATH}/data/protein_input_sets.csv')
amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

def get_individual_ppl(seq, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Tokenize
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].squeeze(0)

    # Apply masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = True,
        mlm_probability = 0.15,
        return_tensors = 'pt'
    )
    
    batch = data_collator([{
        'input_ids': input_ids,
        'attention_mask': inputs['attention_mask'].squeeze(0)
    }])
    
    # Detect which residues were masked
    labels = batch["labels"].squeeze(0)
    mask_positions = labels != -100   

    # Decode tokens back to residues
    residues = tokenizer.convert_ids_to_tokens(input_ids)

    df = pd.DataFrame({
        "pos": list(range(1, len(residues) + 1)),
        "residue": residues,
        "mask": mask_positions.tolist()
    })

    # Move to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Get loss
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss.item()  # This is THIS protein's individual loss
        ppl = math.exp(loss)
    
    return ppl, df 

# Compute perplexity for each sequence
perplexities = []
for idx, protein_id in enumerate(tqdm(proteins_df['ProteinID'].unique(), desc="Computing perplexities")):
    seq = proteins_df[proteins_df['ProteinID'] == protein_id]['Sequence'].iloc[0]
    ppl, mask_df = get_individual_ppl(seq, seed=42 + idx)
    disorder_profile = extract_IDR_profile(seq)
    mask_df_no_pad = mask_df[~mask_df['residue'].isin(['<cls>', '<eos>'])]

    if len(disorder_profile) != len(mask_df_no_pad):
        print(f"Length mismatch for {protein_id}: seq len {len(seq)}, disorder_profile len {len(disorder_profile)}, mask_df_no_pad len {len(mask_df_no_pad)}")
        continue  # Skip this protein due to length mismatch    

    mask_df_no_pad['Disorder'] = disorder_profile
    masked_residues = mask_df_no_pad[mask_df_no_pad['mask'] == True]
    mean_masked_disorder = masked_residues['Disorder'].mean() 

    df = pd.DataFrame({'seq': seq, 'ProteinID' : protein_id, 'ppl' : ppl, 'mean_masked_residue_disorder' : mean_masked_disorder, 'seq_len' : len(seq)}, index = [0])
    
    for aa in amino_acids:
        df[aa] = masked_residues['residue'].value_counts().get(aa, 0)

    perplexities.append(df)

# Save results
pd.concat(perplexities).to_csv(f"{BASE_PATH}/data/esm2_perplexities.csv", index = False)


# ----------------------
# Linear Regression 
#----------------------

ppls = pd.concat(perplexities).rename(columns={'mean_masked_residue_disorder': 'Mean Masked Residue Disorder', 'seq_len': 'Seq Len'})
X = ppls['Mean Masked Residue Disorder'] 
y = ppls['ppl'] 
X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HC1")
model.summary()
conf = model.conf_int(alpha=0.05) 
conf.columns = ['CI_lower', 'CI_upper']

coef_df = pd.DataFrame({
    'variable': model.params.index,
    'beta': model.params.values,
    'p_value': model.pvalues.values, 
    'r2' : model.rsquared,
    't_stat' : model.tvalues

}).join(conf, on='variable')

coef_df[['beta', 'CI_lower', 'CI_upper']] = coef_df[['beta', 'CI_lower', 'CI_upper']].round(2)
coef_df['p_value'] = coef_df['p_value'].apply(lambda p: f"{p:.1e}")

beta_std = model.params['Mean Masked Residue Disorder'] * (X['Mean Masked Residue Disorder'].std() / y.std())

print(coef_df)
print(beta_std)