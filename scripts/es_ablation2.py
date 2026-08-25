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
from scipy.stats import gaussian_kde

'''
Script to Ablate Heads of ESM-2 likely to be associated with electrostatics and 
evaluate ablation effect on MLP Rg prediction 

'''

# -------------------------------------------------------
# Constants
# -------------------------------------------------------

OVERWRITE_NUM_HEADS = 29
MAX_HEADS_ABLATE = 50
SEED             = 42
DROPOUT, HIDDEN_DIM, REPR_LAYER = 0.3, 512, 33
SEQ_LENGTH, TOKENS_PER_BATCH, EMB_DIM = 1022, 4096, 1280
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

print('running aa attention analysis per head/layer normalized by occurrence')
attn = run_aa_analysis_head_layer(normalize = 'occurrence', model = 'original')
attn_occur_norm = attn.groupby(['layer', 'head', 'aa'])['attn_occur_norm'].mean().reset_index()
attn_occur_norm = attn_occur_norm[attn_occur_norm['aa'].isin(['D', 'E', 'K', 'R'])]
attn_occur_norm.to_csv(f'{BASE_PATH}/data/attn_corr_norm_es.csv')
#attn_occur_norm = pd.read_csv(f'{BASE_PATH}/data/attn_corr_norm_es.csv')

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
    by_layer = {}
    for layer_idx, head_idx in heads_to_ablate:
        by_layer.setdefault(layer_idx, []).append(head_idx)
    handles = []
    for layer_idx, head_indices in by_layer.items():
        def make_hook(hidxs):
            def hook(module, input, output):
                out = output[0].clone()
                L, B, D = out.shape
                out = out.view(L, B, num_heads, head_dim)
                for h in hidxs:
                    out[:, :, h, :] = 0.0
                return (out.view(L, B, D),) + output[1:]
            return hook
        handles.append(model.layers[layer_idx].self_attn.register_forward_hook(make_hook(head_indices)))
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


blue_scale_cmap = LinearSegmentedColormap.from_list('white_blue', ['white', '#2367B0'])
def plot_comparison_color(x, y, xlabel, ylabel, title, save_path=None, alpha=0.3, s=20,
                           color=None, color_title='', figsize=(1.65, 1.53), dpi=300):
    x = np.asarray(x); y = np.asarray(y); color = np.asarray(color)
    abs_max = np.abs(color).max()
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='6%', pad=0.05)   
    sc = ax.scatter(x, y, c=color, cmap=blue_scale_cmap, vmin=0, vmax=abs_max,alpha=alpha, s=s, edgecolors='none')
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(color_title, fontsize=3.8)
    cbar.ax.tick_params(labelsize=3.2, pad=1)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle='--', color='red', linewidth=0.7)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(xlabel, fontsize=3.8, linespacing=1.1, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=3.8, linespacing=1.1, labelpad=2)
    ax.tick_params(labelsize=3.2, pad=1)
    ax.set_title(title, fontsize=4.2, pad=2)
    ax.set_box_aspect(1)
    fig.subplots_adjust(left=0.22, right=0.72, top=0.86, bottom=0.24)

    if save_path:
        fig.savefig(save_path, dpi=dpi)

    return fig, ax


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
    base_emb_df   = get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence')
    rg_base       = predict_rg(base_emb_df, N)
    base_spearman, base_pearson, base_mae = evaluate(rg_base, rg_true)
    print(f"Baseline | Spearman={base_spearman:.3f} | Pearson={base_pearson:.3f} | MAE={base_mae:.4f}")

    rng       = np.random.default_rng(SEED)
    all_heads = [(l, h) for l in range(len(esm_model.layers)) for h in range(NUM_HEADS)]
    N_RANDOM_RUNS = 25

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
            rg_abl_neg = predict_rg(get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence'), N)
        finally:
            remove_hooks(handles)

        # positive (cumulative by design)
        heads_pos = list(zip(heads_pos_ranked['layer'][:n_heads], heads_pos_ranked['head'][:n_heads]))
        handles = register_esm_ablation_hooks(esm_model, heads_pos, NUM_HEADS, HEAD_DIM)
        try:
            rg_abl_pos = predict_rg(get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence'), N)
        finally:
            remove_hooks(handles)

        # random: each run ablates its first n_heads from its pre-shuffled sequence (cumulative)
        rand_spearman_runs, rand_pearson_runs = [], []
        for run_idx in range(N_RANDOM_RUNS):
            rand_heads = random_head_sequences[run_idx][:n_heads]
            handles = register_esm_ablation_hooks(esm_model, rand_heads, NUM_HEADS, HEAD_DIM)
            try:
                rg_rand = predict_rg(get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence'), N)
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
    base_emb_df = get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence')
    rg_base     = predict_rg(base_emb_df, N)
    base_spearman, base_pearson, base_mae = evaluate(rg_base, rg_true)
    print(f"Baseline | Spearman={base_spearman:.3f} | Pearson={base_pearson:.3f} | MAE={base_mae:.4f}")

    handles = register_esm_ablation_hooks(esm_model, heads_final, NUM_HEADS, HEAD_DIM)
    try:
        rg_abl = predict_rg(get_esm2_embeddings(proteins, esm_model, alphabet, device, 'sequence'), N)
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

    r_s_base, r_p_base, mae_base = evaluate(results_df['rg_baseline_scaled'], results_df['rg_true_scaled'])
    r_s_abl, r_p_abl, mae_abl = evaluate(results_df['rg_ablated_scaled'],  results_df['rg_true_scaled'])
    print(f"Baseline Flory Norm | Spearman={r_s_base:.3f} | Pearson={r_p_base:.3f} | MAE={mae_base:.4f}")
    print(f"Ablated Flory Norm | Spearman={r_s_abl:.3f} | Pearson={r_p_abl:.3f} | MAE={mae_abl:.4f}")

    t_stat, p_val = stats.ttest_1samp(results_df['delta_mae'], 0)
    print(f"MAE increase: {results_df['delta_mae'].mean():+.3f} ± {results_df['delta_mae'].std():.3f} | t={t_stat:.3f}, p={p_val:.4f}")
    n_worse  = (results_df['delta_mae'] > 0).sum()
    n_better = (results_df['delta_mae'] < 0).sum()
    print(f"Worse: {n_worse} ({100*n_worse/len(results_df):.1f}%) | Better: {n_better} ({100*n_better/len(results_df):.1f}%)")

    color_col = 'frac_KR' if title == 'pos' else 'frac_DE'
    color_title = 'Frac KR' if title == 'pos' else 'Frac DE'
    plot_comparison_color(
        results_df['rg_baseline_scaled'], results_df['rg_ablated_scaled'],
        'Baseline Predicted Rg (Flory Normalized)', 'Ablated Predicted Rg (Flory Normalized)',
        f'{label.upper()} - {title.upper()}', save_path=f'{BASE_PATH}/figures/es_{title}_{label}_scatter.svg',
        alpha=1, s=30, color=results_df[color_col], color_title=color_title)

    return results_df



# -------------------------------------------------------
# Run
# -------------------------------------------------------

idrome_proteins, idrome_true_col = load_idrome()
saxs_proteins,   saxs_true_col   = load_saxs()
heads_final = list(zip(heads_neg_ranked['layer'][:OVERWRITE_NUM_HEADS], heads_neg_ranked['head'][:OVERWRITE_NUM_HEADS]))

run_sweep(idrome_proteins, idrome_true_col, 'idrome', heads_neg_ranked, heads_pos_ranked)
saxs_results_df = run_eval(saxs_proteins, saxs_true_col, heads_final, label = 'saxs', title = 'neg')
results_df = run_eval(idrome_proteins, idrome_true_col, heads_final, label = 'idrome', title = 'neg')
saxs_results_df.to_csv(f'{BASE_PATH}/data/figure_data/saxs_results_df.csv', index = False)
results_df.to_csv(f'{BASE_PATH}/data/figure_data/results_df.csv', index = False)

### Analysis ####
     
results_df['bigger_or_smaller'] = results_df['rg_ablated_scaled'] - results_df['rg_baseline_scaled']
results_df['frac_DE_quartile'] = pd.qcut(results_df['frac_DE'], q=10, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])

deciles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
blue_colors = ['#dbeafe', '#bfdbfe', '#93c5fd', '#60a5fa', '#3b82f6','#2563eb', '#1d4ed8', '#1e40af', '#1e3a8a', '#172554']
decile_color_map = dict(zip(deciles, blue_colors))

def plot_kde_by_decile(ax, data, field, xlabel, color_field='frac_DE_quartile'):
    for q in deciles:
        vals = data.loc[data[color_field] == q, field].dropna().values
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, kde(xs), color=decile_color_map[q], linewidth=0.8, label=q)
    ax.set_xlabel(xlabel, fontsize=4.2, labelpad=2)
    ax.set_ylabel('Density', fontsize=4.2, labelpad=2)
    ax.tick_params(labelsize=3.2, pad=1)

fig, axes = plt.subplots(2, 1, dpi=300)
plot_kde_by_decile(axes[0], results_df, 'bigger_or_smaller', 'Ablated Pred Rg -\nBaseline Pred Rg')
axes[0].axvline(0, color='red', linewidth=1.0)
plot_kde_by_decile(axes[1], results_df, 'rg_baseline_scaled', 'Rg Baseline')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Frac DE\nDecile', fontsize=3.0, title_fontsize=3.2,loc='center left', bbox_to_anchor=(0.83, 0.5), frameon=False, handlelength=0.8,handletextpad=0.3, labelspacing=0.25, borderaxespad=0)
fig.suptitle('Negative Head Ablation - IDRome', fontsize=4.6, y=0.99)
fig.subplots_adjust(left=0.20, right=0.82, top=0.94, bottom=0.10, hspace=0.42)
plt.savefig(f'{BASE_PATH}/figures/es_ablation_kde.svg', dpi=300)
plt.show()

q10 = results_df.loc[results_df['frac_DE_quartile'] == 'Q10', 'delta_mae'].values
rows = []
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']:
    qx = results_df.loc[results_df['frac_DE_quartile'] == q, 'delta_mae'].values
    ks_stat, p_val = ks_2samp(qx, q10)
    cohens_d = cohen_d(q10, qx)
    rows.append({
        'Comparison': f'{q} vs Q10',
        'mean_delta (Q)': round(np.mean(qx), 5),
        'mean_delta (Q10)': round(np.mean(q10), 5),
        'KS': round(ks_stat, 3),
        'p': p_val, 
        'cohen_d' : cohens_d})

ks_df = pd.DataFrame(rows)
reject, p_corrected, _, _ = multipletests(ks_df['p'], method='bonferroni')
ks_df['p_corrected'] = p_corrected.round(4)
ks_df['significant'] = reject
print(ks_df.to_string(index=False))

