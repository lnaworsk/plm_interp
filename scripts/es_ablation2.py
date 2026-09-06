import os, warnings
import torch, numpy as np, pandas as pd
import esm
from esm import FastaBatchedDataset
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
from attention_functions import *
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 
from tqdm import tqdm
import statsmodels.api as sm

'''
Script to Ablate Heads of ESM-2 likely to be associated with electrostatics and 
evaluate ablation effect on MLP Rg prediction 

'''

# -------------------------------------------------------
# Constants
# -------------------------------------------------------

OVERWRITE_NUM_HEADS = 50
MAX_HEADS_ABLATE = 50
SEED             = 42
DROPOUT, HIDDEN_DIM, REPR_LAYER = 0.3, 512, 33
SEQ_LENGTH, TOKENS_PER_BATCH, EMB_DIM = 1022, 8192, 1280
nu_fit, R0_fit   = 0.533, 2.420
coeffs           = np.array([5.18129864, -31.57633081, 63.38468187])
baseline_poly    = np.poly1d(coeffs)
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------------------------------
# Models
# -------------------------------------------------------

class RgMLP(nn.Module):
    def __init__(self, input_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

print("Loading ESM-2...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(device).eval()
HEAD_DIM  = esm_model.embed_dim // esm_model.layers[0].self_attn.num_heads
NUM_HEADS = esm_model.layers[0].self_attn.num_heads
print(f"ESM-2 | embed_dim={esm_model.embed_dim} | num_heads={NUM_HEADS} | head_dim={HEAD_DIM}")

print("Loading Rg MLP...")
rg_model = RgMLP().to(device)
rg_model.load_state_dict(torch.load(os.path.join(BASE_PATH, 'mlp_final.pt'), map_location=device))
rg_model.eval()

# -------------------------------------------------------
# Data loading
# -------------------------------------------------------
def load_idrome():
    supp_table = pd.read_excel(f'{BASE_PATH}/data/Supplementary_Table_3.xlsx')
    df_tesei   = pd.read_csv("https://raw.githubusercontent.com/KULL-Centre/_2023_Tesei_IDRome/main/IDRome_DB.csv")
    df_tesei['Rg_A'] = df_tesei['Rg/nm'] * 10
    df_tesei   = df_tesei.merge(supp_table[['seq_name', 'sequence']], on='seq_name').drop_duplicates()

    seq_to_cluster = {}
    with open(f"{BASE_PATH}/data/idrome_clustered.fasta.clstr") as f:
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

    df_tesei['cluster'] = df_tesei['seq_name'].map(seq_to_cluster)
    df_tesei = df_tesei[df_tesei['cluster'].notna()].reset_index(drop=True)
    print(f"IDRome sequences: {len(df_tesei)} | Clusters: {df_tesei['cluster'].nunique()}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    _, test_idx = next(gss.split(df_tesei['seq_name'].tolist(), groups=df_tesei['cluster'].values))
    proteins = df_tesei.iloc[test_idx].rename(columns={'N': 'seqlen'}).reset_index(drop=True)
    print(f"Test set: {len(proteins)} sequences")
    return proteins, 'Rg_A'

def load_saxs():
    proteins = pd.read_csv(f'{BASE_PATH}/data/all_comparison_data_WITH_STARTING.csv').reset_index(drop=True)
    proteins.columns = proteins.columns.str.strip()
    proteins['seqlen'] = proteins['sequence'].apply(len)
    proteins = proteins.groupby('sequence', as_index=False)['saxs'].mean().reset_index(drop=True)
    proteins['seqlen'] = proteins['sequence'].apply(len)
    print(f"SAXS proteins: {len(proteins)}")
    return proteins, 'saxs'
    
# -------------------------------------------------------
# Head selection (largest only)
# -------------------------------------------------------

# print('running aa attention analysis per head/layer normalized by occurrence')
# attn = run_aa_analysis_head_layer(normalize = 'occurrence', model = 'original')
# attn_occur_norm = attn.groupby(['layer', 'head', 'aa'])['attn_occur_norm'].mean().reset_index()
# attn_occur_norm = attn_occur_norm[attn_occur_norm['aa'].isin(['D', 'E', 'K', 'R'])]
# attn_occur_norm.to_csv(f'{BASE_PATH}/data/attn_corr_norm_es.csv')
attn_occur_norm = pd.read_csv(f'{BASE_PATH}/data/attn_corr_norm_es.csv')

mean_pos = (attn_occur_norm[attn_occur_norm['aa'].isin(['R', 'K'])].groupby(['head', 'layer'])['attn_occur_norm'].mean().reset_index())
mean_neg = (attn_occur_norm[attn_occur_norm['aa'].isin(['D', 'E'])].groupby(['head', 'layer'])['attn_occur_norm'].mean().reset_index())

heads_pos_ranked = mean_pos.sort_values('attn_occur_norm', ascending=False).reset_index(drop=True)
heads_neg_ranked = mean_neg.sort_values('attn_occur_norm', ascending=False).reset_index(drop=True)

# check attention heatmap patterns
vmin = attn_occur_norm['attn_occur_norm'].min()
vmax = attn_occur_norm['attn_occur_norm'].max()
fig, axes = plt.subplots(2, 2, dpi=300)
axes_flat = axes.flatten()
im = None
for i, ax in enumerate(axes_flat):
    aa = attn_occur_norm.aa.unique()[i]
    sub = attn_occur_norm[attn_occur_norm['aa'] == aa]
    pivot = sub.pivot(index='head', columns='layer', values='attn_occur_norm').sort_index(ascending=True)
    im = ax.imshow(pivot.values, aspect='auto', cmap='Blues', vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(aa, fontsize=6)
    ax.set_xlabel('Layer', fontsize=4.5, labelpad=2)
    ax.set_ylabel('Head', fontsize=4.5, labelpad=2)
    ax.tick_params(labelsize=3.2, pad=1)
    step_l = max(1, len(pivot.columns) // 6)
    ax.set_xticks(range(0, len(pivot.columns), step_l))
    ax.set_xticklabels(pivot.columns[::step_l], fontsize=3)
    step_h = max(1, len(pivot.index) // 5)
    ax.set_yticks(range(0, len(pivot.index), step_h))
    ax.set_yticklabels(pivot.index[::step_h], fontsize=3)

fig.suptitle('Mean(Summed Attention/Occurrence)', fontsize=7, y=0.99)
fig.subplots_adjust(left=0.12, right=0.85, top=0.93, bottom=0.06, hspace=0.62, wspace=0.20)
cax = fig.add_axes([0.88, 0.15, 0.03, 0.65])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=4)
plt.savefig(f"{BASE_PATH}/figures/DEKR_heatmaps.svg", dpi=300)
plt.show()

# check correlations of heatmaps 
spread = attn_occur_norm.pivot(columns='aa', index=['layer', 'head']).reset_index()
corr_DE = round(np.corrcoef(spread[('attn_occur_norm', 'D')], spread[('attn_occur_norm', 'E')])[0,1],3)
corr_KR = np.corrcoef(spread[('attn_occur_norm', 'R')], spread[('attn_occur_norm', 'K')])[0,1]

corr_DK = np.corrcoef(spread[('attn_occur_norm', 'D')], spread[('attn_occur_norm', 'K')])[0,1]
corr_DR = np.corrcoef(spread[('attn_occur_norm', 'D')], spread[('attn_occur_norm', 'R')])[0,1]
corr_KE = np.corrcoef(spread[('attn_occur_norm', 'K')], spread[('attn_occur_norm', 'E')])[0,1]
corr_RE = np.corrcoef(spread[('attn_occur_norm', 'R')], spread[('attn_occur_norm', 'E')])[0,1]

print(f'Corr DE {corr_DE}')
print(f'Corr KR {corr_KR}')

print(f'Corr DK {corr_DK}')
print(f'Corr DR {corr_DR}')
print(f'Corr KE {corr_KE}')
print(f'Corr RE {corr_RE}')

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def to_rg(pred_resid, n):
    return pred_resid + baseline_poly(np.log(n))

def register_esm_ablation_hooks(model, heads_to_ablate, num_heads=20, head_dim=64):
    """
    True zero-ablation of ESM-2 attention heads: Att_h(x) -> 0.

    Hooks out_proj's INPUT (register_forward_pre_hook), which is the
    concatenated per-head output [Att_0(x), ..., Att_19(x)] BEFORE
    out_proj mixes heads together. Zeroing the h-th 64-dim slice here
    sets Att_h(x) = 0 while leaving every other head untouched, and
    out_proj then runs normally on the modified input.
    """
    by_layer = {}
    for layer_idx, head_idx in heads_to_ablate:
        by_layer.setdefault(layer_idx, []).append(head_idx)
    handles = []
    for layer_idx, head_indices in by_layer.items():
        def make_pre_hook(hidxs):
            def pre_hook(module, args):
                x = args[0].clone()  # (tgt_len, bsz, embed_dim), pre-out_proj
                for h in hidxs:
                    start, end = h * head_dim, (h + 1) * head_dim
                    x[:, :, start:end] = 0.0
                return (x,) + args[1:]
            return pre_hook
        handles.append(
            model.layers[layer_idx].self_attn.out_proj.register_forward_pre_hook(
                make_pre_hook(head_indices)
            )
        )
    return handles

def remove_hooks(handles):
    for h in handles:
        h.remove()

def predict_rg(emb_df, seqlen):
    emb_cols = [c for c in emb_df.columns if isinstance(c, int)]
    X = torch.tensor(emb_df[emb_cols].values.astype(np.float32)).to(device)
    with torch.no_grad():
        resid = rg_model(X).cpu().numpy()
    N  = seqlen
    rg = np.array([to_rg(p, n) for p, n in zip(resid, N)])
    return rg / (R0_fit * (N ** nu_fit))

def evaluate(pred, true):
    r_s, _ = spearmanr(true, pred)
    r_p, _ = pearsonr(true, pred)
    mae    = np.abs(pred - true).mean()
    return r_s, r_p, mae

def cohen_d(distr1, distr2): 
    mean1 = np.mean(distr1)
    mean2 = np.mean(distr2)
    n1 = len(distr1)
    n2 = len(distr2)
    std1 = np.std(distr1, ddof = 1)
    std2 = np.std(distr2, ddof = 1)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    return cohens_d

# -------------------------------------------------------
# Main sweep + evaluation function
# -------------------------------------------------------

def run_sweep(proteins, true_col, label, heads_neg_ranked, heads_pos_ranked):
    N       = proteins['seqlen'].values
    rg_true = proteins[true_col].values / (R0_fit * (N ** nu_fit))

    # get baseline 
    print(f"\n[{label}] Getting baseline embeddings...")
    base_emb_df   = get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH)
    rg_base       = predict_rg(base_emb_df, N)
    base_spearman, base_pearson, base_mae = evaluate(rg_base, rg_true)
    print(f"Baseline | Spearman={base_spearman:.3f} | Pearson={base_pearson:.3f} | MAE={base_mae:.4f}")

    rng       = np.random.default_rng(SEED)
    all_heads = [(l, h) for l in range(len(esm_model.layers)) for h in range(NUM_HEADS)]
    N_RANDOM_RUNS = 5

    # Pre-build cumulative random head sequences for all 25 runs
    # Each run is a list of length MAX_HEADS_ABLATE, where entry i is the head added at step i+1
    targeted_set = set(
        list(zip(heads_neg_ranked['layer'][:MAX_HEADS_ABLATE], heads_neg_ranked['head'][:MAX_HEADS_ABLATE])) +
        list(zip(heads_pos_ranked['layer'][:MAX_HEADS_ABLATE], heads_pos_ranked['head'][:MAX_HEADS_ABLATE]))
    )
    candidate_heads = [h for h in all_heads if h not in targeted_set]

    random_head_sequences = []  # shape: [N_RANDOM_RUNS][MAX_HEADS_ABLATE]
    for _ in range(N_RANDOM_RUNS):
        perm = [candidate_heads[i] for i in rng.choice(len(candidate_heads), size=MAX_HEADS_ABLATE, replace=False)]
        random_head_sequences.append(perm)

    sweep_results = []
    for n_heads in tqdm(range(1, MAX_HEADS_ABLATE + 1), desc=f'[{label}] Head sweep'):
        
        # negative (cumulative by design)
        heads_neg = list(zip(heads_neg_ranked['layer'][:n_heads], heads_neg_ranked['head'][:n_heads]))
        handles = register_esm_ablation_hooks(esm_model, heads_neg, NUM_HEADS, HEAD_DIM)
        try:
            rg_abl_neg = predict_rg(get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH), N)
        finally:
            remove_hooks(handles)

        # positive (cumulative by design)
        heads_pos = list(zip(heads_pos_ranked['layer'][:n_heads], heads_pos_ranked['head'][:n_heads]))
        handles = register_esm_ablation_hooks(esm_model, heads_pos, NUM_HEADS, HEAD_DIM)
        try:
            rg_abl_pos = predict_rg(get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH), N)
        finally:
            remove_hooks(handles)

        # random: each run ablates its first n_heads from its pre-shuffled sequence (cumulative)
        rand_spearman_runs, rand_pearson_runs = [], []
        for run_idx in range(N_RANDOM_RUNS):
            rand_heads = random_head_sequences[run_idx][:n_heads]
            handles = register_esm_ablation_hooks(esm_model, rand_heads, NUM_HEADS, HEAD_DIM)
            try:
                rg_rand = predict_rg(get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH), N)
            finally:
                remove_hooks(handles)
            r_s_rand, r_p_rand, _ = evaluate(rg_rand, rg_true)
            rand_spearman_runs.append(r_s_rand)
            rand_pearson_runs.append(r_p_rand)

        r_s_rand_mean = np.mean(rand_spearman_runs)
        r_p_rand_mean = np.mean(rand_pearson_runs)
        r_s_rand_std  = np.std(rand_spearman_runs)
        r_p_rand_std  = np.std(rand_pearson_runs)

        r_s_neg, r_p_neg, _ = evaluate(rg_abl_neg,  rg_true)
        r_s_pos, r_p_pos, _ = evaluate(rg_abl_pos,  rg_true)

        sweep_results.append({
            'n_heads': n_heads,
            'delta_spearman_neg': r_s_neg - base_spearman,
            'delta_spearman_pos': r_s_pos - base_spearman,
            'delta_spearman_random': r_s_rand_mean - base_spearman,
            'delta_spearman_random_std': r_s_rand_std,
            'delta_pearson_random_std': r_p_rand_std,
            'delta_pearson_neg': r_p_neg - base_pearson,
            'delta_pearson_pos': r_p_pos - base_pearson,
            'delta_pearson_random': r_p_rand_mean - base_pearson,
        })

        if n_heads % 10 == 0:
            sweep_df = pd.DataFrame(sweep_results)
            sweep_df.to_csv(f'{BASE_PATH}/data/figure_data/es_ablation_sweep_df_{label}_checkpoint_random_consecutive.csv', index=False)
            print(f"Checkpoint saved at n_heads={n_heads}")
            
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df['delta_spearman_random_se'] = sweep_df['delta_spearman_random_std'] / np.sqrt(N_RANDOM_RUNS)
    sweep_df.to_csv(f'{BASE_PATH}/data/figure_data/es_ablation_sweep_df_{label}_random_consecutive.csv', index = False)
    
    sweep_long = sweep_df.melt('n_heads', value_vars=['delta_spearman_neg', 'delta_spearman_pos', 'delta_spearman_random'], var_name='condition', value_name='delta')
    sweep_long['ablation'] = sweep_long['condition'].map({'delta_spearman_neg': 'Negative (D/E)','delta_spearman_pos': 'Positive (K/R)','delta_spearman_random': 'Random'})
    std_df = sweep_df[['n_heads', 'delta_spearman_random_se']].rename(columns={'delta_spearman_random_se': 'se'})
    sweep_long = sweep_long.merge(std_df, on='n_heads', how='left')
    sweep_long['se']         = sweep_long['se'].where(sweep_long['ablation'] == 'Random', 0)
    sweep_long['delta_upper'] = sweep_long['delta'] + sweep_long['se']
    sweep_long['delta_lower'] = sweep_long['delta'] - sweep_long['se']
 
    color_scale = {'Negative (D/E)': '#E8300C','Positive (K/R)': '#2367B0','Random': '#999999'}
    fig, ax = plt.subplots(dpi=300)
    # --- band: shaded confidence region, Random ablation only ---
    random_sub = sweep_long[sweep_long['ablation'] == 'Random'].sort_values('n_heads')
    ax.fill_between(random_sub['n_heads'], random_sub['delta_lower'], random_sub['delta_upper'],color=color_scale['Random'], alpha=0.2, linewidth=0)
    # --- lines + points, one pass per ablation category ---
    for ablation, color in color_scale.items():
        sub = sweep_long[sweep_long['ablation'] == ablation].sort_values('n_heads')
        ax.plot(sub['n_heads'], sub['delta'], color=color, linewidth=1.2, label=ablation)
        ax.scatter(sub['n_heads'], sub['delta'], color=color, s=8, zorder=3, edgecolors='none')
    ax.set_xlabel('Number of Heads Ablated', fontsize=5.5)
    ax.set_ylabel('Δ(Ablated − Baseline) \n Corr(Pred & True Rg) ', fontsize=5.5, linespacing=1.3)
    ax.set_title(f'Head Ablation Sweep', fontsize=6.5)
    ax.tick_params(labelsize=4.5)
    ax.legend(title='Ablation', fontsize=5, title_fontsize=5.5, frameon=False,bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig.subplots_adjust(left=0.16, right=0.72, top=0.78, bottom=0.32)
    plt.savefig(f'{BASE_PATH}/figures/head_ablation_sweep_.svg', dpi=300)
    plt.show()

    print(f"Saved: figures/head_ablation_sweep_.svg")
    return sweep_df

# -------------------------------------------------------
#  evaluation on targeted number of heads 
# -------------------------------------------------------

def run_eval(proteins, true_col, heads_final, label = 'saxs', title = 'neg'):
    N       = proteins['seqlen'].values
    rg_true = proteins[true_col].values / (R0_fit * (N ** nu_fit))

    print(f"\n{[label]} Getting baseline embeddings...")
    base_emb_df = get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH)
    rg_base     = predict_rg(base_emb_df, N)
    base_spearman, base_pearson, base_mae = evaluate(rg_base, rg_true)
    print(f"Baseline | Spearman={base_spearman:.3f} | Pearson={base_pearson:.3f} | MAE={base_mae:.4f}")

    handles = register_esm_ablation_hooks(esm_model, heads_final, NUM_HEADS, HEAD_DIM)
    try:
        abl_emb_df = get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH)
        rg_abl = predict_rg(abl_emb_df, N)
    finally:
        remove_hooks(handles)

    results_df = proteins.copy()
    results_df['rg_true_scaled']     = rg_true
    results_df['rg_baseline_scaled'] = rg_base
    results_df['rg_ablated_scaled']  = rg_abl
    results_df['frac_DE']   = results_df['sequence'].str.count('[DE]') / results_df['sequence'].str.len() * 100
    results_df['frac_KR']   = results_df['sequence'].str.count('[KR]') / results_df['sequence'].str.len() * 100
    results_df['delta_mae'] = (np.abs(results_df['rg_ablated_scaled'] - results_df['rg_true_scaled']) -
                                np.abs(results_df['rg_baseline_scaled'] - results_df['rg_true_scaled']))
    results_df['bigger_or_smaller'] = results_df['rg_ablated_scaled'] - results_df['rg_baseline_scaled']

    r_s_base, r_p_base, mae_base = evaluate(results_df['rg_baseline_scaled'], results_df['rg_true_scaled'])
    r_s_abl, r_p_abl, mae_abl = evaluate(results_df['rg_ablated_scaled'],  results_df['rg_true_scaled'])
    print(f"Baseline Flory Norm | Spearman={r_s_base:.3f} | Pearson={r_p_base:.3f} | MAE={mae_base:.4f}")
    print(f"Ablated Flory Norm | Spearman={r_s_abl:.3f} | Pearson={r_p_abl:.3f} | MAE={mae_abl:.4f}")

    t_stat, p_val = stats.ttest_1samp(results_df['delta_mae'], 0)
    print(f"MAE increase: {results_df['delta_mae'].mean():+.3f} ± {results_df['delta_mae'].std():.3f} | t={t_stat:.3f}, p={p_val:.4f}")
    n_worse  = (results_df['delta_mae'] > 0).sum()
    n_better = (results_df['delta_mae'] < 0).sum()
    print(f"Worse: {n_worse} ({100*n_worse/len(results_df):.1f}%) | Better: {n_better} ({100*n_better/len(results_df):.1f}%)")

    return results_df



# -------------------------------------------------------
# Run
# -------------------------------------------------------

idrome_proteins, idrome_true_col = load_idrome()
saxs_proteins,   saxs_true_col   = load_saxs()

heads_final = list(zip(heads_neg_ranked['layer'][:OVERWRITE_NUM_HEADS], heads_neg_ranked['head'][:OVERWRITE_NUM_HEADS]))
heads_final_pos = list(zip(heads_pos_ranked['layer'][:OVERWRITE_NUM_HEADS], heads_pos_ranked['head'][:OVERWRITE_NUM_HEADS]))

#run_sweep(idrome_proteins, idrome_true_col, 'idrome', heads_neg_ranked, heads_pos_ranked)

# #negative snapshot
saxs_results_df = run_eval(saxs_proteins, saxs_true_col, heads_final, label = 'saxs', title = 'neg')
results_df = run_eval(idrome_proteins, idrome_true_col, heads_final, label = 'idrome', title = 'neg')
saxs_results_df.to_csv(f'{BASE_PATH}/data/figure_data/saxs_results_df.csv', index = False)
results_df.to_csv(f'{BASE_PATH}/data/figure_data/results_df.csv', index = False)

# #positive snapshot
saxs_results_df_pos = run_eval(saxs_proteins, saxs_true_col, heads_final_pos, label = 'saxs', title = 'pos')
results_df_pos = run_eval(idrome_proteins, idrome_true_col, heads_final_pos, label = 'idrome', title = 'pos')
saxs_results_df_pos.to_csv(f'{BASE_PATH}/data/figure_data/saxs_results_df_pos.csv', index = False)
results_df_pos.to_csv(f'{BASE_PATH}/data/figure_data/results_df_pos.csv', index = False)

######## residualization of DEKR 

# baseline embedding ~ DEKR --> residual baseline embedding 
# ablated embedding ~ DEKR --> residual ablated embedding 
# residual baseline embedding  --> residual baseline rg prediction 
# residual ablated embedding --> residual ablated rg prediction 
# ablated embedding --> ablated rg prediction 
# baseline embedding --> baseline rg prediction 
# residual baseline - residual ablated --> any gap between them is NOT attributed to DEKR so has to come from some other electrostatics 

def bootstrap_gap_test(rg_true, pred_base_res, pred_abl_res, n_boot=5000, seed=SEED, method='pearson'):
    """Paired bootstrap over proteins. Returns observed gap, 95% CI, and a
    two-sided bootstrap p-value."""
    corr_fn = pearsonr if method == 'pearson' else spearmanr
    rg_true = np.asarray(rg_true)
    pred_base_res = np.asarray(pred_base_res)
    pred_abl_res = np.asarray(pred_abl_res)
    n = len(rg_true)

    observed_gap = corr_fn(rg_true, pred_abl_res)[0] - corr_fn(rg_true, pred_base_res)[0]

    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)  # same resample applied to both predictions -- paired
        r_abl = corr_fn(rg_true[idx], pred_abl_res[idx])[0]
        r_base = corr_fn(rg_true[idx], pred_base_res[idx])[0]
        gaps[i] = r_abl - r_base

    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])

    if observed_gap < 0:
        p_boot = 2 * (gaps > 0).mean()
    else:
        p_boot = 2 * (gaps < 0).mean()
    p_boot = min(p_boot, 1.0)

    return observed_gap, ci_lo, ci_hi, p_boot

def run_eval_residual(proteins, true_col, heads_final, label = 'saxs', title = 'neg'):
    print(f'running {title}')
    proteins = proteins.copy()
    proteins['frac_DE']   = proteins['sequence'].str.count('[DE]') / proteins['sequence'].str.len() * 100
    proteins['frac_KR']   = proteins['sequence'].str.count('[KR]') / proteins['sequence'].str.len() * 100
    frac_col = 'frac_KR' if title =='pos' else 'frac_DE'
    N       = proteins['seqlen'].values
    rg_true = proteins[true_col].values / (R0_fit * (N ** nu_fit))

    ######### get embeddings 
    print(f"\n{[label]} Getting baseline embeddings...")
    #baseline 
    base_emb_df = get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH)

    #ablated 
    handles = register_esm_ablation_hooks(esm_model, heads_final, NUM_HEADS, HEAD_DIM)
    try:
        abl_emb_df = get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence', tokens_per_batch=TOKENS_PER_BATCH)
    finally:
        remove_hooks(handles)

    ######## residualize embeddings
    # baseline 
    int_cols = [c for c in base_emb_df.columns if isinstance(c, int)]
    X = sm.add_constant(proteins[[f'{frac_col}']])
    y = base_emb_df[int_cols]
    model = sm.OLS(y, X).fit()
    residual_baseline_embeddings = model.resid

    # ablated
    int_cols = [c for c in abl_emb_df.columns if isinstance(c, int)]
    X = sm.add_constant(proteins[[f'{frac_col}']])
    y = abl_emb_df[int_cols]
    model = sm.OLS(y, X).fit()
    residual_abl_embeddings = model.resid

    ######### predict rg 

    # baseline
    rg_base = predict_rg(base_emb_df[int_cols], N)
    base_spearman, base_pearson, base_mae = evaluate(rg_base, rg_true)
    print(f"Baseline | Spearman={base_spearman:.3f} | Pearson={base_pearson:.3f} | MAE={base_mae:.4f}")

    # baseline residual
    rg_base_res = predict_rg(residual_baseline_embeddings, N)
    base_res_spearman, base_res_pearson, base_res_mae = evaluate(rg_base_res, rg_true)
    print(f"Baseline Res | Spearman={base_res_spearman:.3f} | Pearson={base_res_pearson:.3f} | MAE={base_res_mae:.4f}")

    # ablated
    rg_abl = predict_rg(abl_emb_df[int_cols], N)
    abl_spearman, abl_pearson, abl_mae = evaluate(rg_abl, rg_true)
    print(f"Abl | Spearman={abl_spearman:.3f} | Pearson={abl_pearson:.3f} | MAE={abl_mae:.4f}")

    # ablated residual
    rg_abl_res = predict_rg(residual_abl_embeddings, N)
    abl_res_spearman, abl_res_pearson, abl_res_mae = evaluate(rg_abl_res, rg_true)
    print(f"Abl Res | Spearman={abl_res_spearman:.3f} | Pearson={abl_res_pearson:.3f} | MAE={abl_res_mae:.4f}")

    ##### analyse 
    print(f'BEYOND COMPOSITION GAP: Ablated Residual - Baseline Residual: Spearman - {abl_res_spearman - base_res_spearman}, Pearson {abl_res_pearson - base_res_pearson}')

    ##### significance test
    for method in ['pearson', 'spearman']:
        gap, ci_lo, ci_hi, p = bootstrap_gap_test(rg_true, rg_base_res, rg_abl_res, method=method)
        print(f"[{title}] {method} bootstrap: gap={gap:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}], p={p:.4g}")

    return ''

run_eval_residual(idrome_proteins, 'Rg_A', heads_final, label = 'idrome', title = 'neg')
run_eval_residual(idrome_proteins, 'Rg_A', heads_final_pos, label = 'idrome', title = 'pos')

