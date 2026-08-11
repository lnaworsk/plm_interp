import os, warnings
import torch, numpy as np, pandas as pd
import esm
from esm import FastaBatchedDataset
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from scipy import stats
import altair as alt
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
from attention_functions import *
warnings.filterwarnings('ignore')
alt.data_transformers.enable("vegafusion")
alt.themes.register('publication', publication_theme)
alt.themes.enable('publication')
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 
from tqdm import tqdm

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
heatmaps = alt.Chart(attn_occur_norm).mark_rect().encode(
    x=alt.X('layer:O', title='Layer'),
    y=alt.Y('head:O', title='Head'),
    tooltip=['head', 'layer', 'attn_occur_norm'],
    color=alt.Color('attn_occur_norm:Q', title='', scale=alt.Scale())
).properties(height=250, width=360).facet(facet=alt.Facet('aa:N',header=alt.Header(title=None)),columns=2, title = 'Mean(Summed Attention/Occurrence)')
heatmaps.save(f'{BASE_PATH}/figures/DEKR_heatmaps.svg')

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

def plot_comparison_color(x, y, xlabel, ylabel, title, save_path=None, alpha=0.3, s=30, color=None, color_title=''):
    df = pd.DataFrame({xlabel: x, ylabel: y, 'color': color})
    abs_max = df['color'].abs().max()
    lim     = [min(df[xlabel].min(), df[ylabel].min()), max(df[xlabel].max(), df[ylabel].max())]
    scatter = alt.Chart(df).mark_point(opacity=alpha, size=s, filled=True).encode(
        x=alt.X(xlabel, scale=alt.Scale(domain=lim)),
        y=alt.Y(ylabel, scale=alt.Scale(domain=lim)),
        color=alt.Color('color', title=color_title,
                        scale=alt.Scale(domain=[0, abs_max], range=['white', '#2367B0'])))
    line = alt.Chart(pd.DataFrame({xlabel: lim, ylabel: lim})).mark_line(
        strokeDash=[5,5], color='red').encode(x=xlabel, y=ylabel)
    chart = (scatter + line).properties(title=title, width=300, height=300)
    if save_path:
        chart.save(save_path)
    return chart

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
    sweep_df.to_csv(f'{BASE_PATH}/data/figure_data/es_ablation_sweep_df_{label}_random_consecutive.csv', index = False)
    sweep_df['delta_spearman_random_se'] = sweep_df['delta_spearman_random_std'] / np.sqrt(N_RANDOM_RUNS)
    
    sweep_long = sweep_df.melt(
        'n_heads',
        value_vars=['delta_spearman_neg', 'delta_spearman_pos', 'delta_spearman_random'],
        var_name='condition', value_name='delta')
 
    sweep_long['ablation'] = sweep_long['condition'].map({
        'delta_spearman_neg':    'Negative (D/E)',
        'delta_spearman_pos':    'Positive (K/R)',
        'delta_spearman_random': 'Random'
    })
 
    # merge std for random band
    std_df     = sweep_df[['n_heads', 'delta_spearman_random_se']].rename(
        columns={'delta_spearman_random_se': 'se'})
    sweep_long = sweep_long.merge(std_df, on='n_heads', how='left')
    sweep_long['se']         = sweep_long['se'].where(sweep_long['ablation'] == 'Random', 0)
    sweep_long['delta_upper'] = sweep_long['delta'] + sweep_long['se']
    sweep_long['delta_lower'] = sweep_long['delta'] - sweep_long['se']
 
    color_scale = alt.Scale(
        domain=['Negative (D/E)', 'Positive (K/R)', 'Random'],
        range=['#E8300C', '#2367B0', '#999999'])
 
    band = alt.Chart(sweep_long[sweep_long['ablation'] == 'Random']).mark_area(opacity=0.2).encode(
        x='n_heads:Q',
        y='delta_lower:Q',
        y2='delta_upper:Q',
        color=alt.Color('ablation:N', scale=color_scale))
 
    lines = alt.Chart(sweep_long).mark_line().encode(
        x=alt.X('n_heads:Q', title='Number of Heads Ablated'),
        y=alt.Y('delta:Q', title=['Change in Spearman Correlation',
                                   'of Predicted & True Rg (Ablated − Baseline)']),
        color=alt.Color('ablation:N', title='Ablation', scale=color_scale))
 
    points = alt.Chart(sweep_long).mark_point(size=50, filled=True).encode(
        x='n_heads:Q',
        y='delta:Q',
        color=alt.Color('ablation:N', scale=color_scale))
 
    sweep_chart = (band + lines + points).properties(
        width=500, height=350,
        title=f'Head Ablation Sweep — {label.upper()}')
    sweep_chart.save(os.path.join(BASE_PATH, f'figures/es_sweep_{label}_random_consecutive.svg'))
    print(f"Saved: figures/es_sweep_{label}.svg")
 
    return sweep_df, sweep_chart

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
    plot = plot_comparison_color(
        results_df['rg_baseline_scaled'], results_df['rg_ablated_scaled'],
        'Baseline Predicted Rg (Flory Normalized)', 'Ablated Predicted Rg (Flory Normalized)',
        f'{label.upper()} - {title.upper()}', save_path=f'{BASE_PATH}/figures/es_{title}_{label}_scatter.svg',
        alpha=1, s=30, color=results_df[color_col], color_title=color_title)

    return results_df, plot



# -------------------------------------------------------
# Run
# -------------------------------------------------------

idrome_proteins, idrome_true_col = load_idrome()
saxs_proteins,   saxs_true_col   = load_saxs()
heads_final = list(zip(heads_neg_ranked['layer'][:OVERWRITE_NUM_HEADS], heads_neg_ranked['head'][:OVERWRITE_NUM_HEADS]))

run_sweep(idrome_proteins, idrome_true_col, 'idrome', heads_neg_ranked, heads_pos_ranked)
saxs_results_df, neg_eval_saxs = run_eval(saxs_proteins, saxs_true_col, heads_final, label = 'saxs', title = 'neg')
results_df, neg_eval_idrome = run_eval(idrome_proteins, idrome_true_col, heads_final, label = 'idrome', title = 'neg')
saxs_results_df.to_csv(f'{BASE_PATH}/data/figure_data/saxs_results_df.csv', index = False)
results_df.to_csv(f'{BASE_PATH}/data/figure_data/results_df.csv', index = False)


### Analysis ####

     
results_df['bigger_or_smaller'] = results_df['rg_ablated_scaled'] - results_df['rg_baseline_scaled']
results_df['frac_DE_quartile'] = pd.qcut(results_df['frac_DE'], q=10, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])

vline = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='red', strokeWidth=2).encode(x='x:Q')

blue_scale = alt.Scale(
    domain=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'],
    range=['#dbeafe', '#bfdbfe', '#93c5fd', '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#1e40af', '#1e3a8a', '#172554']
)

def kde_chart(data, field, title, color_field='frac_DE_quartile', width=600, height=120):
    return alt.Chart(data).transform_density(
        field,
        as_=[field, 'density'],
        groupby=[color_field]
    ).mark_line().encode(
        alt.X(f'{field}:Q', title=title),
        alt.Y('density:Q', title='Density'),
        alt.Color(f'{color_field}:N', title='Frac DE Decile', sort=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'], scale=blue_scale)
    ).properties(width=width, height=height)

p1 = alt.layer(kde_chart(results_df, 'bigger_or_smaller', 'Ablated Pred Rg - Baseline Pred Rg'), vline)
p2 = kde_chart(results_df, 'rg_baseline_scaled', 'Rg Baseline')
kde_plot = alt.vconcat(p1, p2, spacing=20).properties(title=alt.TitleParams('Negative Head Ablation - IDRome', dy=-14))
kde_plot.save(f'{BASE_PATH}/figures/es_ablation_kde.svg')

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

