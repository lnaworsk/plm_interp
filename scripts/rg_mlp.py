import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import esm
from esm import FastaBatchedDataset
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
from attention_functions import get_esm2_embeddings
BASE_PATH = SCRIPT_DIR.parent 

'''
Script to train Rg prediction model and evaluate on IDRome MD-derived Rg and experimental SAXS Rg 


Data: 
    MD-Derived IDRome Rg: Literature - https://pubmed.ncbi.nlm.nih.gov/38297118/
                          GitHub - "https://raw.githubusercontent.com/KULL-Centre/_2023_Tesei_IDRome/main/IDRome_DB.csv"

    SAXS Experimental Rg: Literature - https://www.nature.com/articles/s41586-026-10141-2
                          Data/Github - https://github.com/holehouse-lab/supportingdata/blob/master/2026/starling_2026/analysis/experimental_comparison/saxs_rg/all_comparison_data.csv

'''

# ---- Constants ------------------------------------------------
SEED             = 42
BATCH_SIZE       = 256
EPOCHS           = 200
LR               = 1e-3
PATIENCE         = 20
WEIGHT_DECAY     = 1e-4
DROPOUT          = 0.3
HIDDEN_DIM       = 512
REPR_LAYER       = 33
SEQ_LENGTH       = 1022
TOKENS_PER_BATCH = 4096
EMB_DIM          = 1280
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

#---- Functions --------------------------------------------------------

def plot_comparison_color(ax, x, y, xlabel, ylabel, title, alpha=0.3, s=30,
                           color=None, color_title='', cbar_ax=None):
    x = np.asarray(x); y = np.asarray(y)
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())

    if color is not None:
        color = np.asarray(color)
        abs_max = np.abs(color).max()
        sc = ax.scatter(x, y, c=color, cmap='RdBu', vmin=-abs_max, vmax=abs_max,
                          alpha=alpha, s=s, edgecolors='none')
        if cbar_ax is not None:
            cbar = plt.colorbar(sc, cax=cbar_ax)
            cbar.set_label(color_title, fontsize=5.5)
            cbar.ax.tick_params(labelsize=4.5)
    else:
        ax.scatter(x, y, color='steelblue', alpha=alpha, s=s, edgecolors='none')

    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle='--', color='red', linewidth=1)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(xlabel, fontsize=5.5)
    ax.set_ylabel(ylabel, fontsize=5.5)
    ax.tick_params(labelsize=4.5)
    if isinstance(title, (list, tuple)):
        ax.set_title('\n'.join(title), fontsize=6, linespacing=1.3)
    else:
        ax.set_title(title, fontsize=6)
    ax.set_box_aspect(1)

def to_target(name):
    """Raw Rg --> residual (training target, in Å)."""
    return rg_lookup[name] - baseline(np.log(n_lookup[name]))

def to_rg(pred_resid, n):
    """Residual --> raw Rg (for evaluation)."""
    return pred_resid + baseline(np.log(n))
    
# ---- Model ----------------------------------------------------------------
class RgMLP(nn.Module):
    """
    MLP on mean-pooled ESM-2 embeddings (1280-dim).
    Predicts scaled Rg residual (Rg minus length baseline).
    Architecture: 1280  --> 512 -->  256 --> 128 --> 1
    """
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

#---- Dataset ----------------------------------------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)

# ----------------------------------------------------------------------------
# PART 1: IDRome — data, embeddings, training
# ----------------------------------------------------------------------------

print("\n" + "="*60)
print("ESM-2 Embedding MLP — Rg prediction ")
print("="*60)

# ---- Load data ----------------------------------------------------------------
supp_table = pd.read_excel(f'{BASE_PATH}/data/Supplementary_Table_3.xlsx')
supp_table['seqlen'] = supp_table['sequence'].apply(len)

df_tesei  = pd.read_csv("https://raw.githubusercontent.com/KULL-Centre/_2023_Tesei_IDRome/main/IDRome_DB.csv")
df_tesei['Rg_A'] = df_tesei['Rg/nm'] * 10
df_tesei  = df_tesei.merge(supp_table[['seq_name', 'sequence']], on='seq_name').drop_duplicates().dropna()
print(f"IDRome sequences: {len(df_tesei)}")

with open(f'{BASE_PATH}/data/idrome.fasta', 'w') as f:
    for _, row in df_tesei.iterrows():
        f.write(f">{row['seq_name']}\n{row['sequence']}\n")

# run externally: cd-hit -i idrome.fasta -o idrome_clustered.fasta -c 0.4 -n 2
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

print(f"Sequences in clustered file: {len(seq_to_cluster)}")
df_tesei['cluster'] = df_tesei['seq_name'].map(seq_to_cluster)
unmapped = df_tesei['cluster'].isna()
print(f"Unmapped sequences: {unmapped.sum()} / {len(df_tesei)}")
rg_lookup = df_tesei.set_index('seq_name')['Rg_A'].to_dict()
n_lookup  = df_tesei.set_index('seq_name')['N'].to_dict()
all_names = df_tesei['seq_name'].tolist()
group_labels = df_tesei['cluster'].values

# ---- Split -------------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(gss.split(all_names, groups=group_labels))

train_val_names  = [all_names[i] for i in train_val_idx]
train_val_groups = group_labels[train_val_idx]
test_names       = [all_names[i] for i in test_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(gss2.split(train_val_names, groups=train_val_groups))

train_names = [train_val_names[i] for i in train_idx]
val_names   = [train_val_names[i] for i in val_idx]
print(f"Train: {len(train_names)} | Val: {len(val_names)} | Test: {len(test_names)}")

# ---- Length baseline (fit on train only) --------------------------------------------
train_N  = np.array([n_lookup[s]  for s in train_names], dtype=float)
train_Rg = np.array([rg_lookup[s] for s in train_names], dtype=float)
coeffs   = np.polyfit(np.log(train_N), train_Rg, deg=2)
baseline = np.poly1d(coeffs)
print(coeffs)

# ---- Fit Flory exponent from training data -----------------------------------
log_N_train  = np.log(train_N)
log_Rg_train = np.log(train_Rg)
nu_fit, log_R0_fit, r_flory_fit, _, _ = linregress(log_N_train, log_Rg_train)
R0_fit = np.exp(log_R0_fit)
print(f"Empirical Flory fit (train) | ν = {nu_fit:.3f} | R0 = {R0_fit:.3f} | r = {r_flory_fit:.3f}")

y_train = np.array([to_target(s) for s in train_names], dtype=np.float32)
y_val   = np.array([to_target(s) for s in val_names],   dtype=np.float32)
y_test  = np.array([to_target(s) for s in test_names],  dtype=np.float32)

# ---- Embeddings --------------------------------------------------------

print("Loading ESM-2...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(device).eval()
df_emb = get_esm2_embeddings(df_tesei, esm_model, alphabet, device, sequence_col='sequence')


emb_cols    = [c for c in df_emb.columns if isinstance(c, int)]
X_all       = df_emb[emb_cols].values.astype(np.float32)
name_to_idx = {n: i for i, n in enumerate(df_emb['seq_name'].tolist())}

X_train = X_all[[name_to_idx[s] for s in train_names]]
X_val   = X_all[[name_to_idx[s] for s in val_names]]
X_test  = X_all[[name_to_idx[s] for s in test_names]]
print(f"Embedding matrix: {X_all.shape}")

# ---- DataLoaders ------------------------------------------------------------
loader_kw    = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)

train_loader = DataLoader(EmbeddingDataset(X_train, y_train), shuffle=True,  **loader_kw)
val_loader   = DataLoader(EmbeddingDataset(X_val,   y_val), shuffle=False, **loader_kw)
test_loader  = DataLoader(EmbeddingDataset(X_test,  y_test), shuffle=False, **loader_kw)

# ---- Training ----------------------------------------------------------------
model      = RgMLP().to(device)
optimiser  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
scaler_amp = torch.cuda.amp.GradScaler()
criterion  = nn.MSELoss()

train_hist, val_hist        = [], []
best_val_loss, patience_ctr = float('inf'), 0

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for X_b, y_b in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimiser.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(X_b), y_b)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimiser)
        scaler_amp.update()
        train_losses.append(loss.item())
    scheduler.step()
    train_hist.append(np.mean(train_losses))

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            with torch.cuda.amp.autocast():
                val_losses.append(criterion(model(X_b), y_b).item())
    val_hist.append(np.mean(val_losses))
    print(f"Epoch {epoch+1:3d} | train: {train_hist[-1]:.4f} "
          f"| val: {val_hist[-1]:.4f}")

    if val_hist[-1] < best_val_loss:
        best_val_loss, patience_ctr = val_hist[-1], 0
        torch.save(model.state_dict(), os.path.join(BASE_PATH, 'mlp_final.pt'))
        print(f" Best val loss: {best_val_loss:.4f}, saved")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break


training_curve_df = pd.DataFrame({"epoch": np.arange(len(train_hist)), "Training": train_hist, "Validation": val_hist })
training_curve_long = training_curve_df.melt(id_vars="epoch", value_vars=["Training", "Validation"], var_name="dataset", value_name="mse_loss")
training_curve_long.to_csv(f'{BASE_PATH}/data/figure_data/training_curve_long.csv')
best_epoch = int(np.argmin(training_curve_long[training_curve_long['dataset']=='Validation'].mse_loss))

dataset_color_map = {'Training': 'steelblue', 'Validation': 'orange'}  
fig, ax = plt.subplots(dpi=300)
for ds, color in dataset_color_map.items():
    sub = training_curve_long[training_curve_long['dataset'] == ds].sort_values('epoch')
    ax.plot(sub['epoch'], sub['mse_loss'], color=color, linewidth=1.2, label=ds)
ax.axvline(best_epoch, color='red', linestyle='--', linewidth=1.0, alpha=0.5)
ax.set_xlabel('Epoch', fontsize=6, labelpad=3)
ax.set_ylabel('MSE Loss', fontsize=6)
ax.set_title('Training vs Validation Loss', fontsize=7)
ax.tick_params(labelsize=5)
ax.legend(fontsize=5.5, frameon=False, loc='upper right')
fig.subplots_adjust(left=0.14, right=0.95, top=0.96, bottom=0.08)
plt.savefig(os.path.join(BASE_PATH, "figures/mlp_final_curves.svg"), dpi=300)
plt.show()

# ---- Evaluate on IDRome test set ------------------------------------------------
model.load_state_dict(torch.load(os.path.join(BASE_PATH, 'mlp_final.pt')))
model.eval()

pred_scaled, true_scaled = [], []
with torch.no_grad():
    for X_b, y_b in tqdm(test_loader, desc='Evaluating'):
        X_b = X_b.to(device)
        with torch.cuda.amp.autocast():
            pred_scaled.extend(model(X_b).cpu().numpy())
        true_scaled.extend(y_b.numpy())

pred_scaled = np.array(pred_scaled)
true_scaled = np.array(true_scaled)
N_test      = np.array([n_lookup[s] for s in test_names])
pred_raw    = np.array([to_rg(p, n) for p, n in zip(pred_scaled, N_test)])
true_raw    = np.array([to_rg(t, n) for t, n in zip(true_scaled, N_test)])

true_raw_check = np.array([rg_lookup[s] for s in test_names])
assert np.allclose(true_raw, true_raw_check), "true_raw does not match rg_lookup!"
print("✓ true_raw exactly matches rg_lookup for test_names")


df_eval = pd.DataFrame({'seq_name': test_names, 'Rg_true':  true_raw, 'Rg_pred':  pred_raw, 'N': N_test})
df_eval['Rg_true_norm'] = df_eval['Rg_true'] / (R0_fit * (df_eval['N'] ** nu_fit))
df_eval['Rg_pred_norm'] = df_eval['Rg_pred'] / (R0_fit * (df_eval['N'] ** nu_fit))
df_eval.to_csv(f'{BASE_PATH}/data/figure_data/df_eval.csv', index = False)

# ---------------------------------------------------------------------------
# PART 2: SAXS validation
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("PART 2: SAXS validation")
print("="*60)

#  '''
#  all_comparison_data.csv : https://github.com/holehouse-lab/supportingdata/blob/master/2026/starling_2026/analysis/experimental_comparison/saxs_rg/all_comparison_data.csv
#  '''

saxs_proteins = pd.read_csv(f'{BASE_PATH}/data/all_comparison_data_WITH_STARTING.csv').reset_index(drop=True)
saxs_proteins = saxs_proteins.rename(columns={' sequence': 'sequence'})
saxs_proteins['seqlen'] = saxs_proteins['sequence'].apply(len)
saxs_proteins = (saxs_proteins.groupby('sequence', as_index=False)[[' saxs', ' starling_mps', ' starling_cuda']].mean().reset_index(drop=True))
saxs_proteins['seqlen'] = saxs_proteins['sequence'].apply(len)
saxs_proteins['saxs_norm'] = saxs_proteins[' saxs'] / (R0_fit * (saxs_proteins['seqlen'] ** nu_fit))
saxs_proteins['starling_mps_norm'] = saxs_proteins[' starling_mps'] / (R0_fit * (saxs_proteins['seqlen'] ** nu_fit))
saxs_proteins['starling_cuda_norm'] = saxs_proteins[' starling_cuda'] / (R0_fit * (saxs_proteins['seqlen'] ** nu_fit))

print(f"Running {len(saxs_proteins)} SAXS proteins")

saxs_emb_df = get_esm2_embeddings(saxs_proteins, esm_model, alphabet, device, sequence_col='sequence')

emb_cols_saxs = [c for c in saxs_emb_df.columns if isinstance(c, int)]
X_saxs        = saxs_emb_df[emb_cols_saxs].values.astype(np.float32)
N_saxs        = saxs_proteins['seqlen'].values

model.eval()
with torch.no_grad():
    pred_saxs_scaled = model(torch.tensor(X_saxs, dtype=torch.float32).to(device)).cpu().numpy()

pred_saxs_raw = np.array([to_rg(p, n) for p, n in zip(pred_saxs_scaled, N_saxs)])
saxs_proteins['Rg_pred'] = pred_saxs_raw
saxs_proteins['Rg_pred_flory_norm'] = saxs_proteins['Rg_pred'] / (R0_fit * (saxs_proteins['seqlen'] ** nu_fit))
saxs_proteins.to_csv(f'{BASE_PATH}/data/figure_data/saxs_proteins.csv', index = False)

# ---------------------------------------------------------------------------
# PART 3: Plotting
# ---------------------------------------------------------------------------

r_idrome, _ = pearsonr(df_eval['Rg_true_norm'], df_eval['Rg_pred_norm'])
r_s_idrome, _ = spearmanr(df_eval['Rg_true_norm'], df_eval['Rg_pred_norm'])
r_flory_saxs, _       = pearsonr(saxs_proteins['saxs_norm'], saxs_proteins['Rg_pred_flory_norm'])
r_s_flory_saxs, _       = spearmanr(saxs_proteins['saxs_norm'], saxs_proteins['Rg_pred_flory_norm'])
r_starling, _ = pearsonr(saxs_proteins['saxs_norm'], saxs_proteins['starling_cuda_norm'])
r_s_starling, _ = spearmanr(saxs_proteins['saxs_norm'], saxs_proteins['starling_cuda_norm'])

fig, axes = plt.subplots(1, 3, dpi=300)
plot_comparison_color(
    axes[0], x=df_eval['Rg_true_norm'], y=df_eval['Rg_pred_norm'],
    xlabel='Tesei Rg\nFlory Normalized (Å)', ylabel='MLP Predicted Rg\nFlory Normalized (Å)',
    title=['IDRome-MLP', f'Pearson r = {r_idrome:.3f}, Spearman r = {r_s_idrome:.3f}'],
    alpha=1, s=8)

plot_comparison_color(
    axes[1], x=saxs_proteins['saxs_norm'], y=saxs_proteins['Rg_pred_flory_norm'],
    xlabel='Experimental SAXS Rg\nFlory Normalized (Å)', ylabel='Predicted Rg\nFlory Normalized (Å)',
    title=['SAXS-MLP', f'Pearson r = {r_flory_saxs:.3f}, Spearman r = {r_s_flory_saxs:.3f}'],
    alpha=1, s=8)

plot_comparison_color(
    axes[2], x=saxs_proteins['saxs_norm'], y=saxs_proteins['starling_cuda_norm'],
    xlabel='Experimental SAXS Rg\nFlory Normalized (Å)', ylabel='STARLING Cuda Rg\nFlory Normalized (Å)',
    title=['SAXS-STARLING', f'Pearson r = {r_starling:.3f}, Spearman r = {r_s_starling:.3f}'],
    alpha=1, s=8)

fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.23, wspace=0.22)
plt.savefig(f'{BASE_PATH}/figures/rg_mlp.svg', dpi=300)
plt.show()
