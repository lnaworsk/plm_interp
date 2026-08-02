import torch
import pandas as pd
import tempfile
import numpy as np
import re
from attention_functions import attention_rollout, extract_IDR_profile, extract_IDR, get_attn_rollout
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer,AutoModel
import json
import statsmodels.formula.api as smf
from scipy import stats
import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 

# True if you want to use the exact sequences utilized in this study
# False if you want to download the current versions of the external ClinVar data 
derived = True 

'''
Script to Get Disease-Related Variant Positions and assess attention to non-disease relevant residues 

ClinVar accessed on February 2, 2026

External Data Required: 
gene2refseq is from https://ftp.ncbi.nlm.nih.gov/gene/DATA/ 
GRCh38_latest_protein is from https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/
9606_database comes from describprot - https://biomine.cs.vcu.edu/server-handler/?type=servers&target=DESCRIBEPROT/

'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------
#      Functions
# --------------------

# Function to extract WT AA, position, and Mut AA
def parse_mutation(annotation):
    match = re.search(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\)", annotation)
    if match:
        wt_aa, pos, mut_aa = match.groups()
        return wt_aa, int(pos), mut_aa
    return None, None, None

def get_seq_wt(seq, pos):
    if pd.isna(pos):
        return None
    pos = int(pos)  # ensure integer
    if pos <= 0 or pos > len(seq):
        return None
    return seq[pos-1]

###############
# Load Data 
###############

if derived == False: 

    # Get ClinVar Variants
    filepath_summary = f'{BASE_PATH}/data/variant_summary.txt.gz'
    summary = pd.read_csv(filepath_summary, sep='\t', compression='gzip', dtype=str)
    summary = summary[summary['Type'] == 'single nucleotide variant'] # single point mutations only 
    allowed_clnsig = [ "Pathogenic","Likely pathogenic","Pathogenic/Likely pathogenic"] # only pathogenic 
    protein_variants = summary[summary['ClinicalSignificance'].isin(allowed_clnsig)]

    # Get NM to NP 
    # gene2refseq is from https://ftp.ncbi.nlm.nih.gov/gene/DATA/ 

    mapping_cols = ['#tax_id', 'mRNA_acc', 'prot_acc']
    mapping_data = []
    iter_csv = pd.read_csv(f'{BASE_PATH}/data/gene2refseq', sep='\t', usecols=[0, 3, 5], names=mapping_cols, header=0, chunksize=100000)
    for chunk in iter_csv:
        human_chunk = chunk[chunk['#tax_id'] == 9606]
        mapping_data.append(human_chunk)
    mapping_df = pd.concat(mapping_data)

    nm_to_np = dict(zip(mapping_df['mRNA_acc'], mapping_df['prot_acc']))

    # Get NP to Seq 
    #GRCh38_latest_protein is from https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/

    fasta_path = f'{BASE_PATH}/data/GRCh38_latest_protein_copy.faa'
    local_sequences = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        local_sequences[record.id] = str(record.seq)

    protein_variants["NM"] = protein_variants["Name"].str.extract(r'^(N[MP]_\d+\.\d+)')
    protein_variants['NP'] = protein_variants['NM'].map(nm_to_np)
    protein_variants = protein_variants[~protein_variants['NP'].isna()]
    protein_variants['Seq'] = protein_variants['NP'].map(local_sequences)
    protein_variants[~protein_variants['Seq'].isna()]
    protein_variants['Seq'] = protein_variants['Seq'].astype(str)

    # filter by sequence length 
    protein_variants['seqlen'] = protein_variants['Seq'].apply(len)
    protein_variants = protein_variants[protein_variants['seqlen']<1024] # remove pproteins > 1024 in length
    protein_variants = protein_variants[protein_variants['seqlen']>50] # only include proteins > 50 in length 
    protein_variants = protein_variants[~protein_variants['Seq'].str.contains('U')]
    trunc_protein_variants = protein_variants[['Type', 'Name', 'GeneID', 'GeneSymbol', 'ClinicalSignificance', 'NM', 'NP', 'Seq', 'seqlen']].copy()

    # filter to remove genes which had multiple sequences 
    seq_counts = trunc_protein_variants.groupby('GeneSymbol')['Seq'].nunique() #filter for genes that got mapped to more than one sequence
    genes_multiple_sequences = seq_counts[seq_counts > 1]
    trunc_protein_variants = trunc_protein_variants[~trunc_protein_variants['GeneSymbol'].isin(genes_multiple_sequences.index)]

    ### check if the mutations are correct (IE - the sequence we called from NCBI matches the AAs in the Name col)
    trunc_protein_variants[["wt_aa", "pos", "mut_aa"]] = trunc_protein_variants["Name"].apply(lambda x: pd.Series(parse_mutation(x)))

    aa_dict = {
        "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C",
        "Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
        "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P",
        "Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"}

    # Convert to single letter codes
    trunc_protein_variants["wt_aa"] = trunc_protein_variants["wt_aa"].map(aa_dict)
    trunc_protein_variants["mut_aa"] =trunc_protein_variants["mut_aa"].map(aa_dict)
    trunc_protein_variants = trunc_protein_variants[(~trunc_protein_variants['wt_aa'].isna()) & (~trunc_protein_variants['mut_aa'].isna())].drop_duplicates() # removing things like Terine 

    # Extract WT from seq (guard against NaN positions)
    trunc_protein_variants["seq_wt"] = trunc_protein_variants.apply(lambda row: get_seq_wt(row["Seq"], row["pos"]), axis=1)
    test = trunc_protein_variants[trunc_protein_variants['seq_wt']!= trunc_protein_variants['wt_aa']] # filter out the cases where the WT AA does not match the one found at the variant position in the sequence retrieved form NCBI

    if len(test)>1: 
        print('ISSUE: Some sequences do not align')
    trunc_protein_variants.to_csv(f'{BASE_PATH}/data/clinvar_sequences.csv', index = False)


elif derived == True: 
    trunc_protein_variants = pd.read_csv(f'{BASE_PATH}/data/clinvar_sequences.csv')

# ----------------------
# Get Attention Rollout
# ----------------------

model_name = "facebook/esm2_t33_650M_UR50D"
model = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=False, output_attentions=True )
model = model.to(device) 
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, local_files_only=True)
print('Finished loading ESM2')

rollout_df = []
for seq in tqdm(trunc_protein_variants.Seq.unique()):

    idr_profile = extract_IDR_profile(seq)
    slice_df = trunc_protein_variants[trunc_protein_variants['Seq'] == seq]
    slice_df['pos'] = slice_df['pos'].astype(int)
    rollout = get_attn_rollout(seq, device = device, tokenizer = tokenizer, model = model)
    labels = [f"{aa}_{i}" for i, aa in enumerate(seq.strip().upper(), 1)]
    rollout_mini_df = pd.DataFrame({'labels' : labels,'rollout' : rollout, 'seq' : seq, 'idr' : idr_profile})
    rollout_mini_df['pos'] = rollout_mini_df.labels.str.split('_').str[1]
    rollout_mini_df['AA'] = rollout_mini_df.labels.str.split('_').str[0]
    rollout_mini_df['pos'] = rollout_mini_df['pos'].astype(int)
    rollout_mini_df['path_mut'] = rollout_mini_df['pos'].isin(slice_df.pos)
    rollout_df.append(rollout_mini_df)

comb_data = pd.concat(rollout_df)
aa_groups = {
    **{aa: "Hydrophobic" for aa in ['A', 'I', 'L', 'M', 'F', 'V']},
    **{aa: "Polar"       for aa in ['S', 'Q', 'N', 'G', 'C', 'T', 'P']},
    **{aa: "Cation"     for aa in ['K', 'R', 'H']},
    **{aa: "Anion"      for aa in ['D', 'E']},
    **{aa: "Aromatic"   for aa in ['W', 'Y', 'F']},
}
comb_data["AA_group"] = comb_data["AA"].map(aa_groups)
comb_data["idr_perc"] = comb_data.groupby("seq")["idr"].rank(pct=True)
comb_data["attn_perc"] = comb_data.groupby("seq")["rollout"].rank(pct=True)
comb_data['mut_label_desc'] = comb_data['path_mut'].map({True: 'Disease-Relevant', False: 'Non-Disease-Relevant'})
comb_data['idr_perc_bin'] = pd.cut(comb_data['idr_perc'], bins=10, labels=[f'{i}' for i in range(10)])
comb_data.to_csv(f'{BASE_PATH}/data/figure_data/comb_data.csv', index = False)

# ----------------------
# Plotting
# ----------------------

print(f'Number of Unique Sequences: {comb_data.seq.nunique()}')
false_data = comb_data[comb_data['path_mut'] == False][['attn_perc']]
median_false = np.median(false_data)
q1_f, q3_f = np.percentile(false_data, [25, 75])
report = f"{median_false:.2f} ({q1_f:.2f}–{q3_f:.2f})"
print("Non-Disease Relevant Median Attn (IQR):", report)
true_data = comb_data[comb_data['path_mut'] == True][['attn_perc']]
median_true = np.median(true_data)
q1_t, q3_t = np.percentile(true_data, [25, 75])
report = f"{median_true:.2f} ({q1_t:.2f}–{q3_t:.2f})"
print("Disease Relevant Median Attn (IQR):", report)

# Pathogenic v. Non-Pathogenic Attention Distribtutions 
fig, (ax1, ax2) = plt.subplots(1, 2)
# Disease-Relevant
ax1.hist(comb_data[comb_data['path_mut'] == True]['attn_perc'], bins=20, color = 'steelblue')
ax1.set_xlabel('Attention Received Percentile')
ax1.set_ylabel('Count of Residues')
ax1.set_title('Disease-Relevant Positions')
# Non-Disease-Relevant
ax2.hist(comb_data[comb_data['path_mut'] == False]['attn_perc'], bins=20, color='steelblue')
ax2.set_xlabel('Attention Received Percentile')
ax2.set_ylabel('Count of Residues')
ax2.set_title('Non-Disease-Relevant Positions')
fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.28, wspace=0.35)
plt.savefig(f'{BASE_PATH}/figures/clinvar_expanded_attention_percentile_distr.png', dpi=300)
plt.show()


# attention by disorder percentile 
fig, ax = plt.subplots()
bins = sorted(comb_data['idr_perc_bin'].unique())
x = np.arange(len(bins))
width = 0.35
colors = {'Non-Disease-Relevant': 'steelblue', 'Disease-Relevant': 'orange'}
for i, (label, offset) in enumerate(zip(['Non-Disease-Relevant', 'Disease-Relevant'], [-width/2, width/2])):
    data_by_bin = [comb_data[(comb_data['idr_perc_bin'] == b) & (comb_data['mut_label_desc'] == label)]['attn_perc'].dropna()for b in bins]
    bp = ax.boxplot(data_by_bin,positions=x + offset,widths=width * 0.8,patch_artist=True,manage_ticks=False,boxprops=dict(facecolor=colors[label],alpha=0.7,linewidth=0.4)
    ,medianprops=dict(color='black',linewidth=0.6),whiskerprops=dict(linewidth=0.4),capprops=dict(linewidth=0.4),flierprops=dict(marker='o',markersize=2,alpha=0.3))
ax.set_xticks(x)
ax.set_xticklabels(bins)
ax.set_xlabel('Disorder Percentile Bins')
ax.set_ylabel('Attention Percentile')
ax.set_ylim(0, 1)
ax.set_title('Attention Percentile across Degrees of Disorder')
legend_labels = ['Non-Disease-\nRelevant', 'Disease-\nRelevant']
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[l], alpha=0.7) for l in ['Non-Disease-Relevant', 'Disease-Relevant']]
ax.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
          fontsize=5, labelspacing=1.0, handlelength=1.2, handletextpad=0.5)
fig.subplots_adjust(left=0.16, right=0.76, top=0.72, bottom=0.28)
plt.savefig(f'{BASE_PATH}/figures/clinvar_expanded_attention_percentile_by_idr_percentile_bins.png', dpi=300)
plt.show()

# Attention by Grouping
group_order = ['Hydrophobic', 'Polar', 'Cation', 'Anion', 'Aromatic']
colors = {'Non-Disease-Relevant': 'steelblue', 'Disease-Relevant': 'orange'}
labels = ['Non-Disease-Relevant', 'Disease-Relevant']
width = 0.35
group_sizes = [len(set(aa for aa, g in aa_groups.items() if g == grp)) for grp in group_order]

fig, axes = plt.subplots(1, len(group_order), sharey=True,gridspec_kw={'width_ratios': group_sizes, 'wspace': 0.12})
for i, g in enumerate(group_order):
    ax = axes[i]
    data = comb_data[comb_data['AA_group'] == g]
    aas = sorted(data.groupby('AA')['attn_perc'].median().sort_values(ascending=False).index)
    x = np.arange(len(aas))

    for j, (label, offset) in enumerate(zip(labels, [-width/2, width/2])):
        data_by_aa = [data[(data['AA'] == aa) & (data['mut_label_desc'] == label)]['attn_perc'].dropna()for aa in aas]
        ax.boxplot(data_by_aa,
                   positions=x + offset,
                   widths=width * 0.8,
                   patch_artist=True,
                   manage_ticks=False,
                   boxprops=dict(facecolor=colors[label], alpha=0.7, linewidth=0.4),
                   medianprops=dict(color='black', linewidth=0.6),
                   whiskerprops=dict(linewidth=0.4),
                   capprops=dict(linewidth=0.4),
                   flierprops=dict(marker='o', markersize=1, alpha=0.3))

    ax.set_xticks(x)
    ax.set_xticklabels(aas, fontsize=4.5)
    ax.set_xlabel('')
    ax.set_title(g, fontsize=5.5)
    ax.tick_params(axis='y', labelsize=4.5)
    if i == 0:
        ax.set_ylabel('Attention Percentile', fontsize=5.5)

handles = [plt.Rectangle((0,0),1,1, facecolor=colors[l], alpha=0.7) for l in labels]
fig.legend(handles, [l.replace('-', '-\n') for l in labels], loc='center left',
           bbox_to_anchor=(0.905, 0.5), fontsize=4.5, labelspacing=0.8,
           handlelength=1.0, handletextpad=0.4, frameon=False)
fig.subplots_adjust(left=0.06, right=0.90, top=0.66, bottom=0.20)
plt.savefig(f'{BASE_PATH}/figures/clinvar_expanded_attn_percentile_by_aa_group.png', dpi=300)
plt.show()

# AA Composition of Pathogenic WT 

mut_aa_counts = comb_data[comb_data['path_mut']==True].groupby('AA').size().reset_index()
mut_aa_counts.columns = ['AA', 'count']
mut_aa_counts = mut_aa_counts.sort_values('count', ascending=False)
fig, ax = plt.subplots()
ax.bar(mut_aa_counts['AA'], mut_aa_counts['count'], color = 'steelblue')
ax.set_xlabel('Amino Acid Type')
ax.set_ylabel('Count')
ax.set_title('Amino Acid Counts of\nDisease-Relevant WT Residues')
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/figures/clinvar_expanded_disease_aa_count.png', bbox_inches='tight')
plt.show()

# ------------------------------------
# Evolutionary Conservation Comparison
# ------------------------------------
# the 9606_database comes from describprot - https://biomine.cs.vcu.edu/server-handler/?type=servers&target=DESCRIBEPROT/

clinvar_sequences = trunc_protein_variants.Seq.unique()
results = []
with open(f"{BASE_PATH}/data/9606_database.json", "r") as f:
    for line in f:
        try:
            line = line.strip().strip(',').strip('[').strip(']')
            if not line: continue
            
            data = json.loads(line)
            full_seq = data.get('seq')

            if full_seq in clinvar_sequences:
                raw_cons_str = data.get('MMseq2_conservation_score', "")
                
                # Correctly handle the string conversion once per sequence
                if raw_cons_str:
                    cons_list = [
                        float(x) if x.strip().upper() != 'NULL' else np.nan 
                        for x in raw_cons_str.split(',')
                    ]
                else:
                    cons_list = [np.nan] * len(full_seq)
                
                # Iterate through the sequence to create per-residue rows
                for i, aa in enumerate(full_seq):
                    results.append({
                        'seq': full_seq,
                        'pos': i + 1,
                        'AA': aa,
                        'conservation': cons_list[i] if i < len(cons_list) else np.nan
                    })
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

conservation_df = pd.DataFrame(results)

# Data Preparation
reg_data = comb_data.merge(conservation_df, on=['seq', 'pos', 'AA']).drop_duplicates()
print(f'comb_data.seq.nunique() = {comb_data.seq.nunique()}')
print(f'reg_data.seq.nunique()  = {reg_data.seq.nunique()}')

reg_data['path_mut'] = reg_data['path_mut'].astype('category')
reg_data['path_mut_int'] = reg_data['path_mut'].cat.codes
reg_data['conservation'] = pd.to_numeric(reg_data['conservation'], errors='coerce')

bins = [0, 0.25, 0.75, 1.0]
labels = ["0 - 0.25)", "0.25 - 0.75)", "0.75 - 1.0"]
reg_data['idr_bin'] = pd.cut(reg_data['idr'], bins=bins, labels=labels, right=False)

report_ols = []
report_log =[]
for i in labels: 
    reg_data_bin = reg_data[reg_data['idr_bin'] == i]
    reg_data_bin['rollout_z'] = (reg_data_bin['rollout'] - reg_data_bin['rollout'].mean()) / reg_data_bin['rollout'].std()
    reg_data_bin['conservation_z'] = (reg_data_bin['conservation'] - reg_data_bin['conservation'].mean()) / reg_data_bin['conservation'].std()

    cluster_kwds = {'cov_type': 'cluster', 'cov_kwds': {'groups': reg_data_bin['seq']}}

    # OLS: rollout ~ path_mut + conservation
    # Decomposes how much variance in attention is explained by disease relevance vs conservation
    ols_full        = smf.ols('rollout ~ path_mut + conservation', data=reg_data_bin).fit(**cluster_kwds)
    ols_path_only   = smf.ols('rollout ~ path_mut',                data=reg_data_bin).fit(**cluster_kwds)
    ols_cons_only   = smf.ols('rollout ~ conservation',            data=reg_data_bin).fit(**cluster_kwds)

    partial_r2_path_mut = (ols_full.rsquared - ols_cons_only.rsquared) / (1 - ols_cons_only.rsquared)

    reporting_df_ols = pd.DataFrame({'idr_bin' : i, 'r2_full' : ols_full.rsquared, 'r2_path_mut_only' : ols_path_only.rsquared, 'r2_cons_only' : ols_cons_only.rsquared, 'partial_r2' : partial_r2_path_mut, 'beta_path_mut_unadjusted' : ols_path_only.params.filter(like='path_mut').values[0] , 'beta_path_mut_adjusted' : ols_full.params.filter(like='path_mut').values[0], "n" : len(reg_data_bin)}, index = [0])

    # Logit: path_mut ~ rollout/conservation
    # Tests whether attention predicts disease relevance beyond conservation
    logit_rollout = smf.logit('path_mut_int ~ rollout_z',      data=reg_data_bin).fit(**cluster_kwds)
    logit_cons    = smf.logit('path_mut_int ~ conservation_z', data=reg_data_bin).fit(**cluster_kwds)
    logit_both    = smf.logit('path_mut_int ~ rollout_z + conservation_z', data=reg_data_bin).fit(**cluster_kwds)

    # LR test: does rollout add predictive power beyond conservation?
    lr_stat = -2 * (logit_cons.llf - logit_both.llf)
    lr_pval = stats.chi2.sf(lr_stat, df=1)

    report_df_log = pd.DataFrame({'idr_bin' : i, 
                                'rollout_z coef (unadjusted)' : logit_rollout.params['rollout_z'], 
                                'rollout_z coef (adjusted)' : logit_both.params['rollout_z'], 
                                'conservation_z coef (unadjusted)' : logit_cons.params['conservation_z'], 
                                'conservation_z coef (adjusted)' : logit_both.params['conservation_z'], 
                                'LR_test' : lr_stat,
                                'n' : len(reg_data_bin)}, index = [0])
    report_ols.append(reporting_df_ols)
    report_log.append(report_df_log)


ols_df  = pd.concat(report_ols)
ols_df.to_csv(f'{BASE_PATH}/data/figure_data/ols_df.csv')
log_df  = pd.concat(report_log)
log_df.to_csv(f'{BASE_PATH}/data/figure_data/log_df.csv')

# log df plot 
log_df_melted = log_df[['idr_bin', 'rollout_z coef (adjusted)', 'conservation_z coef (adjusted)', 'LR_test']].rename(columns = {'rollout_z coef (adjusted)': 'Attention Rollout (z)', 'conservation_z coef (adjusted)' : 'Conservation (z)'})
log_df_melted = log_df_melted.melt(id_vars = 'idr_bin')
data = log_df_melted[log_df_melted['variable'] != 'LR_test']
idr_bins = sorted(data['idr_bin'].unique())
variables = data['variable'].unique()
n_bins = len(idr_bins)
x = np.arange(len(variables))
width = 0.6 / len(variables)
color_map = dict(zip(variables, ['orange', 'steelblue']))
fig, axes = plt.subplots(1, n_bins, sharey=True)
for i, idr_bin in enumerate(idr_bins):
    ax = axes[i]
    bin_data = data[data['idr_bin'] == idr_bin]
    for j, var in enumerate(variables):
        val = bin_data[bin_data['variable'] == var]['value'].values
        if len(val) > 0:
            ax.bar(j, val[0], color=color_map[var])
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.set_xlabel('')
    ax.set_title(f'{idr_bin}', fontsize=5.5)
    ax.tick_params(axis='y', labelsize=4.5)
    if i == 0:
        ax.set_ylabel('Adjusted Beta', fontsize=5.5)
fig.suptitle('Disease Relevancy ~\nAttention, Conservation', fontsize=6, y=0.99)
handles = [plt.Rectangle((0,0),1,1, color=color_map[v]) for v in variables]
short_labels = ['Rollout (z)', 'Conservation (z)']
fig.legend(handles, short_labels, loc='center left', bbox_to_anchor=(0.80, 0.5),
           fontsize=4.5, frameon=False, handlelength=1.0, handletextpad=0.4,
           labelspacing=0.8)
fig.subplots_adjust(left=0.12, right=0.79, top=0.70, bottom=0.10, wspace=0.15)
plt.savefig(f'{BASE_PATH}/figures/clinvar_log_reg.png', dpi=300)
plt.show()

# ols df plot 
ols_df_melted = ols_df[['beta_path_mut_adjusted', 'beta_path_mut_unadjusted', 'idr_bin']].melt(id_vars='idr_bin')
ols_df_melted['variable_label'] = ols_df_melted['variable'].str.extract(r'(unadjusted|adjusted)', flags=re.IGNORECASE)[0].str.capitalize()
idr_bins = sorted(ols_df_melted['idr_bin'].unique())
variable_labels = ols_df_melted['variable_label'].unique()

fig, axes = plt.subplots(1, n_bins, sharey=True)
for i, idr_bin in enumerate(idr_bins):
    ax = axes[i]
    bin_data = ols_df_melted[ols_df_melted['idr_bin'] == idr_bin]
    ax.bar(bin_data['variable_label'], bin_data['value'], color = 'steelblue')
    ax.set_title(f'{idr_bin}')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)
    if i == 0:
        ax.set_ylabel('Beta on Disease Relevancy')
fig.suptitle('Attention Rollout ~\nDisease Relevancy, Conservation')
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/figures/ols_clinvar.png')
plt.show()