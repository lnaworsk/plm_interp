import torch
import pandas as pd
import numpy as np
import esm
import joblib
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer,AutoModel
from attention_functions import *
import umap
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold 
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import false_discovery_control
from scipy.stats import pearsonr, norm
from esm.data import FastaBatchedDataset
import tempfile, os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 

'''
Script to compute attention rollout and corresponding attention analysis on ESM2 attention for 
ordered and disordered proteins 

'''

GROUP_COLORS = {'PDB': 'orange', 'DisProt': 'steelblue'}
plt.style.use(f'{BASE_PATH}/publication.mplstyle')


protein_sets = pd.read_csv(f'{BASE_PATH}/data/protein_input_sets.csv')
protein_sets['Group'] = np.where(protein_sets['ProteinID'].str.contains('pdb', case=False, na=False), 'pdb', np.where(protein_sets['ProteinID'].str.contains('disprot', case=False, na=False), 'disprot', None))

print('loading attention rollout data')
attn_rollout = joblib.load(f"{BASE_PATH}/attention_matrices_chunks/attention_rollout_comb_original.joblib")
print('done loading attention rollout data')

# split the dictionary into the disordered and ordered sets 
pdb_dict = {k: v for k, v in attn_rollout.items() if 'pdb' in k.lower()}
disprot_dict = {k: v for k, v in attn_rollout.items() if 'disprot' in k.lower()}

# --------------------------------
# Input Set Seq Len Distributions #
# --------------------------------

protein_lengths = protein_sets.copy()
protein_lengths['seq_len'] = protein_lengths['Sequence'].apply(len)
protein_lengths['Group'] = protein_lengths['Group'].replace({'pdb': 'PDB','disprot': 'DisProt'})

# plotting 
bins = np.histogram_bin_edges(protein_lengths['seq_len'], bins=40)  # shared edges so bars align
fig, ax = plt.subplots(layout='constrained')
for g in ['DisProt', 'PDB']:
    ax.hist(protein_lengths.loc[protein_lengths.Group == g, 'seq_len'], bins=bins, label=g, color=GROUP_COLORS[g], alpha=0.55, edgecolor='none')
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Count")
ax.set_title("PDB & DisProt Sequence Length")
ax.legend(title="Group", loc='upper right')
plt.savefig(f"{BASE_PATH}/figures/input_sets_seqlen_distribution.svg", dpi=300)
plt.show()
print('saved seqlen distribution plots')

# --------------------------------
# UMAP of ESM2 Embeddings #
# --------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False  

model_name = "esm2_t33_650M_UR50D"
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)

esm2_embeddings = get_esm2_embeddings(protein_sets, model, alphabet, device, sequence_col = "Sequence", repr_layer = 33, seq_length =  1022, tokens_per_batch = 4096)
esm2_embeddings['Sequence_length'] = esm2_embeddings['Sequence'].apply(len)
esm2_embeddings['idr'] = esm2_embeddings['Sequence'].apply(extract_IDR)
esm2_embeddings['idr_frac'] = esm2_embeddings['idr']/esm2_embeddings['Sequence_length'] * 100

reducer = umap.UMAP(random_state = 42, n_components = 2, n_neighbors = 15, min_dist = 0.1)
embedding_cols = esm2_embeddings.columns[[str(col).isdigit() for col in esm2_embeddings.columns]]
scaled_data = StandardScaler().fit_transform(esm2_embeddings[embedding_cols])
embedding_umap = reducer.fit_transform(scaled_data)

umap_df = pd.DataFrame({'0' : embedding_umap[:, 0], '1' : embedding_umap[:, 1], 'Group' : esm2_embeddings['Group'], 'Protein_ID' : esm2_embeddings['ProteinID'], 'IDR Frac' : esm2_embeddings['idr_frac']})
umap_df['Group'] = umap_df['Group'].replace({'pdb': 'PDB','disprot': 'DisProt'})
umap_df.to_csv(f'{BASE_PATH}/data/figure_data/umap_df.csv', index = False)

# plotting 
fig, axes = plt.subplots(1, 2, dpi=300)
# --- Left panel ---
ax = axes[0]
for g in ['PDB', 'DisProt']:
    mask = umap_df['Group'] == g
    alpha = 0.05 if g == 'PDB' else 1.0
    ax.scatter(umap_df['0'][mask], umap_df['1'][mask], label=g, alpha=alpha, s=10, color=GROUP_COLORS[g], edgecolors='none', zorder=2 if g == 'PDB' else 1)
ax.set_title("UMAP of ESM-2 Embeddings")
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend(title="Group", bbox_to_anchor=(.75, .75), loc='upper left')
ax.set_box_aspect(1)
# --- Right panel ---
ax = axes[1]
sc = ax.scatter(umap_df['0'], umap_df['1'], c=umap_df['IDR Frac'], cmap='viridis', s=10)
ax.set_title("UMAP of ESM-2 Embeddings")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_box_aspect(1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label("IDR Fraction")

plt.tight_layout()
plt.savefig(f"{BASE_PATH}/figures/umap.svg", dpi=300)
plt.show()
print('saved umap plot')

# Separation in original ESM2 space (1280D)
labels = esm2_embeddings['Group'].map({'pdb': 0, 'disprot': 1})
silhouette_esm2 = silhouette_score(esm2_embeddings[embedding_cols], labels)
print(f"Silhouette (ESM2 1280D): {silhouette_esm2:.3f}")

with open(f"{BASE_PATH}/data/pdb_disprot.fasta", "w") as f:
    for i in protein_sets.ProteinID.unique():
        seq = protein_sets.loc[protein_sets['ProteinID'] == i, 'Sequence'].iloc[0]
        cdhit_id = f"{i}"
        f.write(f">{cdhit_id}\n")
        f.write(f"{seq}\n")

# # Run CD-HIT externally, then continue:
# cd-hit -i pdb_disprot.fasta -o clustered_input.fasta -c 0.40 -n 2

# Parse CD-HIT clusters
seq_to_cluster = {}
with open(f"{BASE_PATH}/data/clustered_input.fasta.clstr") as f:
    current_cluster = None
    for line in f:
        line = line.strip()
        if line.startswith(">Cluster"):
            current_cluster = line.split()[1]
        else:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            seq_id_part = parts[1].strip()
            seq_id = seq_id_part.split()[0].lstrip(">").rstrip("...")
            seq_to_cluster[seq_id] = current_cluster

protein_sets['cluster'] = protein_sets['ProteinID'].map(seq_to_cluster)
esm2_embeddings = esm2_embeddings.merge(protein_sets[['ProteinID', 'cluster']].drop_duplicates(), on='ProteinID', how='left')

X_cv = esm2_embeddings[embedding_cols]
labels = esm2_embeddings['Group'].map({'pdb': 0, 'disprot': 1})
y_cv = labels
groups = esm2_embeddings['cluster'].astype(int)
n_splits = 5
print(f"{n_splits}-fold GroupKFold over {groups.nunique()} clusters")

clf = LogisticRegression(max_iter=5000)
scores = cross_val_score(clf, X_cv, y_cv, groups=groups, cv=GroupKFold(n_splits=n_splits))
print(f'Linear Classifier Accuracy (cluster-grouped CV): {scores.mean():.3f} ± {scores.std():.3f}')

# ---------------------------------------------
# Disorder of Disprot/PDB Protein Input Sets #
# ---------------------------------------------

idr_df = esm2_embeddings[['Group', 'ProteinID', 'idr']]
idr_df['Group'] = idr_df['Group'].replace({'pdb': 'PDB','disprot': 'DisProt'})
idr_df.to_csv(f'{BASE_PATH}/data/figure_data/idr_df.csv', index = False)

# plotting
fig, ax = plt.subplots(dpi=300, layout='constrained')
fig.get_layout_engine().set(w_pad=0.15, h_pad=0.15)  
for g in ['DisProt', 'PDB']:
    vals = idr_df.loc[idr_df['Group'] == g, 'idr'].dropna().values
    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(xs, kde(xs), color=GROUP_COLORS[g], linewidth=1.2, label=g)
ax.set_xlabel('# of Disordered Residues per Sequence', fontsize=5.5)
ax.set_ylabel('Density', fontsize=5.5)
ax.tick_params(labelsize=4.5)
ax.legend(title='Group', fontsize=5, title_fontsize=5.5, frameon=False, loc='upper right')
plt.savefig(f"{BASE_PATH}/figures/input_sets_disorder.svg", dpi=300)
plt.show()

# --------------------------------
# Rollout Distr for Each AA #
# --------------------------------

pdb_distr = get_attention_distribution_df(pdb_dict, normalize = False, standardize = False)
disprot_distr = get_attention_distribution_df(disprot_dict, normalize = False, standardize = False)
disprot_distr['Group'] = 'DisProt'
pdb_distr['Group'] = 'PDB'

summary_stats_disprot = (disprot_distr.groupby(['amino_acid', 'Group'])['attention'].agg(['mean', 'std', 'count']).reset_index())
summary_stats_pdb = (pdb_distr.groupby(['amino_acid', 'Group'])['attention'].agg(['mean', 'std', 'count']).reset_index())
summary_stats_disprot['se'] = summary_stats_disprot['std'] / np.sqrt(summary_stats_disprot['count'])
summary_stats_pdb['se'] = summary_stats_pdb['std'] / np.sqrt(summary_stats_pdb['count'])

cohens_df = cohens_d(pd.concat([pdb_distr, disprot_distr]))
cohens_d_final = cohens_df.sort_values('cohens_d')[['amino_acid','cohens_d', 'p_value_uncorrected', 'n_g1', 'n_g2']]
print('Cohens_d for attention rollout', cohens_d_final)
mean_aa_attn = pd.concat([summary_stats_pdb, summary_stats_disprot])
mean_aa_attn.to_csv(f'{BASE_PATH}/data/figure_data/mean_aa_attn.csv', index = False)

# plotting 
amino_acids = mean_aa_attn["amino_acid"].unique()
x = np.arange(len(amino_acids))
width = 0.8 / 2

fig, ax = plt.subplots(figsize=(7.2, 3.25), dpi=300)
for i, g in enumerate(['PDB', 'DisProt']):
    sub = mean_aa_attn[mean_aa_attn["Group"] == g]
    sub = sub.set_index("amino_acid").reindex(amino_acids)
    offsets = x + (i - 1) * width + width/2
    ax.bar(offsets, sub["mean"], width=width, label=g, yerr=sub["se"], capsize=2,
           linewidth=0.8, color=GROUP_COLORS[g])
ax.set_xticks(x)
ax.set_xticklabels(amino_acids, rotation=0)
ax.set_xlabel("Amino Acid")
ax.set_ylabel("Mean Attention Rollout Received")
ax.set_title("Mean Attention Received per Amino Acid with Standard Error")
ax.legend(title="")
plt.tight_layout()
plt.savefig(f"{BASE_PATH}/figures/mean_aa_attn_received.svg", dpi=300)
plt.show()
print('saved mean aa attention received plot')


# -------------------------------------------------------------
# AA Attn Received @ Each Head/Layer #
# -------------------------------------------------------------

print('running aa attention analysis per head/layer normalized by occurrence')
attn = run_aa_analysis_head_layer(normalize = 'occurrence', model = 'original')
attn['Group'] = np.where(attn['protein'].str.contains('pdb', case=False, na=False), 'PDB', np.where(attn['protein'].str.contains('disprot', case=False, na=False), 'DisProt', None))
attn_occur_norm = attn.groupby(['Group', 'layer', 'head', 'aa'])['attn_occur_norm'].mean().reset_index()
attn_occur_norm[attn_occur_norm['aa'].isin(['C', 'D'])].to_csv(f'{BASE_PATH}/data/figure_data/attn_occur_norm.csv', index = False)
print("Performing statistical tests across all head/layer/AA combinations...")
stats_results = fast_statistical_comparison(attn)

stats_results['p_value_fdr'] = false_discovery_control(stats_results['p_value_uncorrected'])
stats_results['significant'] = stats_results['p_value_fdr'] < 0.05
sig_results = stats_results[stats_results['significant'] == True]

cohens_df = sig_results.groupby('amino_acid')['cohens_d'].describe().reset_index().sort_values('mean').round(3)
print(f'Attn per head/layer cohens df: {cohens_df}')
print(f'Length of Sig Results: {len(sig_results.significant)} out of {len(stats_results.significant)} samples')
del attn # clear up memory 

# plotting 
aa_list = attn_occur_norm['aa'].unique()
batch_size = 3

for i in range(0, len(aa_list), batch_size):
    batch_aas = aa_list[i:i+batch_size]
    fig, axes = plt.subplots(len(batch_aas), 2, figsize=(10, 2.5*len(batch_aas)), dpi=300, squeeze=False)
    for r, aa in enumerate(batch_aas):
        for c, group in enumerate(['PDB', 'DisProt']):
            ax = axes[r, c]
            data = attn_occur_norm[(attn_occur_norm.aa == aa) & (attn_occur_norm.Group == group)]
            heatmap = data.pivot(index='head', columns='layer', values='attn_occur_norm').sort_index()

            im = ax.imshow(heatmap, aspect='auto', origin='lower',
                           cmap='Blues',
                           vmin=heatmap.values.min(),
                           vmax=heatmap.values.max())

            ax.set(title=f'{aa} - {group}', xlabel='Layer', ylabel='Head')
            ax.set_xticks(np.arange(len(heatmap.columns)))
            ax.set_xticklabels(heatmap.columns, rotation=90, fontsize=7)
            ax.set_yticks(np.arange(len(heatmap.index)))
            ax.set_yticklabels(heatmap.index, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}/figures/attn_summed_norm_occur_{i}.svg', bbox_inches='tight')
    plt.close()

print('saved attention summed norm occur head/layer plots')


# --------------------------------
# AA Composition Distributions #
# --------------------------------

disprot_distr['occurrence'] = disprot_distr.groupby(['amino_acid', 'protein_id'])['amino_acid'].transform('count')
pdb_distr['occurrence'] = pdb_distr.groupby(['amino_acid', 'protein_id'])['amino_acid'].transform('count')
occurence_df = pd.concat([pdb_distr, disprot_distr])

# Calculate proportional frequencies
disprot_props = occurence_df[occurence_df['Group']=='DisProt'].groupby('amino_acid')['occurrence'].sum() / occurence_df[occurence_df['Group']=='DisProt']['occurrence'].sum()
pdb_props = occurence_df[occurence_df['Group']=='PDB'].groupby('amino_acid')['occurrence'].sum() / occurence_df[occurence_df['Group']=='PDB']['occurrence'].sum()
disprot_props_df = pd.DataFrame(disprot_props).reset_index()
disprot_props_df['Group'] = 'DisProt'
pdb_props_df = pd.DataFrame(pdb_props).reset_index()
pdb_props_df['Group'] = 'PDB'
comb_props = pd.concat([disprot_props_df, pdb_props_df])
comb_props['occurrence'] = comb_props['occurrence'] * 100  # Convert to percentage
comb_props.to_csv(f'{BASE_PATH}/data/figure_data/comb_props.csv', index = False)

# plotting 
amino_acids = sorted(comb_props['amino_acid'].unique())
x = np.arange(len(amino_acids))
width = 0.8 / 2
fig, ax = plt.subplots(dpi=300, layout='constrained')
fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1)
for i, g in enumerate(['DisProt', 'PDB']):
    sub = comb_props[comb_props['Group'] == g].set_index('amino_acid').reindex(amino_acids)
    offsets = x + (i - 1) * width + width/2
    ax.bar(offsets, sub['occurrence'], width=width, label=g, color=GROUP_COLORS[g])

ax.set_xticks(x)
ax.set_xticklabels(amino_acids, fontsize=5)
ax.set_xlabel('Amino Acid', fontsize=6)
ax.set_ylabel('Proportional\nFrequency (%)', fontsize=6, linespacing=1.1)
ax.tick_params(axis='y', labelsize=5)
ax.legend(title='Group', fontsize=5.5, title_fontsize=6, frameon=False, loc='upper right')
plt.savefig(f"{BASE_PATH}/figures/aa_proportional_frequency.svg", dpi=300)
plt.show()
print('saved aa proportional frequency plot')

# -------------------------------------
# Cor(Disorder, Attn Rollout Received) 
# -------------------------------------

disorder_comp = []
for data_dict, group in [(disprot_dict, 'DisProt'), (pdb_dict, 'PDB')]:
    for protein, data in data_dict.items():
        sequence = data['sequence']

        idr_profile = extract_IDR_profile(sequence)
        tot_idr = extract_IDR(sequence)
        attention_received = data['rollout'].sum(dim=0).cpu().numpy()

        disorder_df = pd.DataFrame({
            'label': [f'{aa}_{i}' for i, aa in enumerate(sequence.strip().upper(), 1)],
            'idr': idr_profile,
            'attn_received': attention_received,
            'tot_idr': tot_idr,
            'seq_len': len(sequence),
            'protein': protein,
            'Group': group,
        })

        disorder_df['corr'] = np.corrcoef(disorder_df['idr'],disorder_df['attn_received'])[0, 1]
        disorder_comp.append(disorder_df)

disorder_comp_concat = pd.concat(disorder_comp, ignore_index=True)
disorder_comp_concat['pos'] = disorder_comp_concat.label.str.split('_').str[1]
disorder_comp_concat['disorder_frac'] = disorder_comp_concat['tot_idr'] / disorder_comp_concat['seq_len'] * 100 
disorder_comp_concat['idr_bin'] = pd.cut(disorder_comp_concat['idr'], bins=4)
disorder_comp_concat['idr_bin_str'] = disorder_comp_concat['idr_bin'].astype(str)
disorder_comp_concat.to_csv(f'{BASE_PATH}/data/figure_data/disorder_comp_concat.csv', index = False)

# plotting 
fig, ax = plt.subplots(figsize=(7.2, 1.53))
x_grid = np.linspace(0, 5, 300)
colors = ['#deebf7', '#9ecae1', '#3182bd', '#08519c']
groups = sorted(disorder_comp_concat['idr_bin_str'].unique())
for i, g in enumerate(groups):
    vals = disorder_comp_concat.loc[disorder_comp_concat['idr_bin_str'] == g, 'attn_received'].dropna().values
    kde = gaussian_kde(vals)
    density = kde(x_grid)
    ax.plot(x_grid, density, linewidth=3, color=colors[i % len(colors)], label=g)
ax.set_xlim(0, 5)
ax.set_yscale('log')
ax.set_xlabel("Attention Rollout Received")
ax.set_ylabel("Probability Density")
ax.set_title("Distribution of Attention by Disorder Probability Bin")

# -------------------------------------------------------------
# TDP 43 Case Study  # and SI Examples 
# -------------------------------------------------------------

print('Loading ESM2 ....')
model_name = "facebook/esm2_t33_650M_UR50D"
model = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=False, output_attentions=True)
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, local_files_only=True)
print('Finished loading ESM2')

seq_list = ['MSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDNKRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVMSQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSLCGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDSKSSGWGM', 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD', 'MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGGRGRGGSGGGGGGGGGGYNRSSGGYEPRGRGGGRGGRGGMGGSDRGGFNKFGGPRDQGSRHDSEQDNSDNNTIFVQGLGENVTIESVADYFKQIGIIKTNKKTGQPMINLYTDRETGKLKGEATVSFDDPPSAKAAIDWFDGKEFSGNPIKVSFATRRADFNRGGGNGRGGRGRGGPMGRGGYGGGGSGGGGRGGFPSGGGGGGGQQRAGDWKCPNPTCENMNFSWRNECNQCKAPKPDGPGGGPGGSHMGGNYGDDRRGGRGGYDRGGYRGRGGDRGGFRGGRGGGDRGGFGPGKMDSRGEHRQDRRERPY', 'MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSENRCHLEGGSDQKDLVQELQEEKPSSSHLVSRPSTSSRRRAISETEENSDELSGERQRKRHKSDSISLSFDESLALCVIREICCERSSSSESTGTPSNPDLDAGVSEHSGDWLDQDSVSDQFSVEFEVESLDSEDYSLSEEGQELSDEDDEVYQVTVYQAGESDTDSFEEDPEISLADYWKCTSCNEMNPPLPSHCNRCWALRENWLPEDKGKDKGEISEKAKLENSTQAEEGFDVPDCKKTIVNDSRESCVEENDDKITQASQSQESEDYSQPSTSSSIIYSSQEDVKEFEREETQDKEESVESSLPLNAIEPCVICQGRPKNGCIVHGKTGHLMACFTCAKKLKKRNKPCPVCRQPIQMIVLTYFP', 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADMEDVCGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQIRLQAEAFQARLKSWFEPLVEDMQRQWAGLVEKVQAAVGTSAAPVPSDNH']
example_seq_df = pd.DataFrame({'ProteinID': ['TDP-43', 'P53', 'FUS', 'MDM2', 'APOE'], 'Sequence': seq_list})

line_color_map = {'Attention Received': 'steelblue','Disorder': 'orange',}

regions = pd.DataFrame({
    'region': ['NTD', 'RRM1', 'RRM2', 'CTD'],
    'start': [1, 104, 191, 274],
    'end': [103, 200, 262, 413],
})
regions['label_pos'] = (regions['start'] + regions['end']) / 2

dfs = []
for seq in example_seq_df.Sequence:
    pid = example_seq_df[example_seq_df['Sequence'] == seq]['ProteinID'].values[0]
    rollout = get_attn_rollout(seq, device=device, tokenizer=tokenizer, model=model)
    idr = extract_IDR_profile(seq)
    labels = [f"{aa}_{i}" for i, aa in enumerate(seq.strip().upper(), 1)]
    df = pd.DataFrame({'label': labels, 'idr': idr, 'attn_received': rollout, 'seq': seq, 'pid': pid})
    df['pos'] = df.label.str.split('_').str[1].astype(int)
    df_altair_input = df.melt(id_vars=['pos'], value_vars=['attn_received', 'idr'])
    df_altair_input["variable"] = np.where(df_altair_input["variable"] == "attn_received", "Attention Received",np.where(df_altair_input["variable"] == "idr", "Disorder", None))
    df_altair_input['pid'] = pid
    dfs.append(df_altair_input)

    if pid == 'TDP-43':
        fig, ax = plt.subplots()
        for i, row in regions.iterrows():
            ax.axvspan(row['start'], row['end'],color=colors[i % len(colors)],alpha=0.2,label=row['region'])

        for var in df_altair_input['variable'].unique():
            sub = df_altair_input[df_altair_input['variable'] == var].sort_values('pos')
            ax.plot(sub['pos'], sub['value'], label=var, linewidth=1.2, color=line_color_map[var])

        y_top = ax.get_ylim()[1]
        for _, row in regions.iterrows():
            ax.text(row['label_pos'], y_top * 1.02, row['region'], ha='center', va='bottom', fontsize=5)

        ax.set_xlabel("Position")
        ax.set_ylabel("Value")
        ax.set_title(f"{pid} Attention Rollout Received vs. Disorder Profile")
        ax.set_ylim(top=y_top * 1.22)

        ax.legend(fontsize=5,frameon=False,bbox_to_anchor=(1.02, 1),loc='upper left',borderaxespad=0)
        fig.subplots_adjust(left=0.09, right=0.80, top=0.72, bottom=0.28)
        plt.savefig(f"{BASE_PATH}/figures/tdp43_idr_attn_received.svg", dpi=300)
        plt.show()
        print('saved tdp43 idr attn received plot')

    del rollout

si_examples = pd.concat(dfs)
si_examples.to_csv(f'{BASE_PATH}/data/figure_data/si_examples.csv', index=False)

# Multi-panel figure for the remaining (non-TDP-43) proteins
pids = [p for p in si_examples['pid'].unique() if p != 'TDP-43']
n = len(pids)

fig, axes = plt.subplots(n, 1, dpi=300)
axes_flat = np.atleast_1d(axes)  

for i, ax in enumerate(axes_flat):
    pid = pids[i]
    sub = si_examples[si_examples['pid'] == pid]

    for var, color in line_color_map.items():
        vsub = sub[sub['variable'] == var].sort_values('pos')
        ax.plot(vsub['pos'], vsub['value'], color=color, linewidth=0.8, label=var)

    ax.set_title(pid, fontsize=5)
    ax.tick_params(labelsize=3.5)
    ax.set_xlabel('Position', fontsize=4)
    ax.set_ylabel('Value', fontsize=4)

handles = [plt.Line2D([0], [0], color=c, linewidth=1.2) for c in line_color_map.values()]
fig.legend(handles, line_color_map.keys(), loc='center left', bbox_to_anchor=(0.885, 0.5),
           fontsize=5, frameon=False, title='Variable', title_fontsize=5.5,
           handlelength=0.9, borderaxespad=0, handletextpad=0.4)

fig.suptitle('SI Examples: Attention Rollout Received vs. Disorder Profile', fontsize=8, y=0.995)
fig.subplots_adjust(left=0.055, right=0.86, top=0.94, bottom=0.05, hspace=0.65, wspace=0.40)
plt.savefig(f"{BASE_PATH}/figures/si_examples.svg", dpi=300)
plt.show()