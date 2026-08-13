import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 
from typing import Tuple
from scipy.stats import gaussian_kde
import re 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr, pearsonr
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy.stats import gaussian_kde

############################
# Helpers & Theme 
############################

plt.style.use(f'{BASE_PATH}/publication.mplstyle')
plt.rcParams['axes.prop_cycle'] = cycler(color=['#56B4E9', '#E69F00', '#CC79A7', '#009E73','#F0E442', '#0072B2', '#D55E00', '#000000'])
mpl.rcParams['svg.fonttype'] = 'none'
group_color_map = {'PDB': 'orange', 'DisProt': 'steelblue'}
# Base dimensions (mm)
CANVAS_W: float = 183.0
CANVAS_H: float = 170.0
GAP: float = 5.0
COLS: int = 4
ROWS: int = 4

UNIT_W: float = (CANVAS_W - (COLS - 1) * GAP) / COLS  # 42.0 mm
UNIT_H: float = (CANVAS_H - (ROWS - 1) * GAP) / ROWS  # 38.75 mm
MM_TO_IN: float = 1 / 25.4

def panel_size(cols: int = 1, rows: int = 1) -> Tuple[float, float]:
    """Return (width, height) in inches for a panel spanning cols x rows grid units."""
    w: float = (cols * UNIT_W + (cols - 1) * GAP) * MM_TO_IN
    h: float = (rows * UNIT_H + (rows - 1) * GAP) * MM_TO_IN
    return round(w, 2), round(h, 2)


# Reference table
for c, r in [(1,1), (2,1), (1,2), (2,2), (3,1), (4,1), (4,2), (4,4)]:
    w, h = panel_size(c, r)
    w_mm = w / MM_TO_IN
    h_mm = h / MM_TO_IN
    print(f"{c}x{r}:  {w_mm:6.1f} x {h_mm:5.1f} mm  ->  fig, ax = plt.subplots(figsize=({w}, {h}))")

#####################################
# Figure 1 - Attention Analysis 
#####################################

###### umap
umap_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/umap_df.csv')

fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.53), dpi=300)
# --- Left panel ---
ax = axes[0]
plot_order = sorted(umap_df['Group'].unique(), key=lambda g: g == 'PDB') # draw DisProt first (opaque, background), then PDB last (transparent, on top)
for g in plot_order:
    mask = umap_df['Group'] == g
    alpha = 0.05 if g == 'PDB' else 1.0
    ax.scatter(umap_df['0'][mask], umap_df['1'][mask], label=g, alpha=alpha, s=5, color=group_color_map[g], edgecolors='none', zorder=2 if g == 'PDB' else 1)
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_box_aspect(1)
# --- Right panel ---
ax = axes[1]
sc = ax.scatter(umap_df['0'], umap_df['1'], c=umap_df['IDR Frac'], cmap='viridis', s=5)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_box_aspect(1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label("IDR Fraction")
plt.tight_layout()
plt.savefig(f"{BASE_PATH}/figures/final/umap.svg", dpi=300)
plt.show()

###### disorder attn kde 
disorder_comp_concat = pd.read_csv(f'{BASE_PATH}/data/figure_data/disorder_comp_concat.csv')
disorder_comp_concat = disorder_comp_concat[disorder_comp_concat['attn_received'] < 5]

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
ax.legend(title="Disorder Probability Bin",bbox_to_anchor=(1.02, 1),loc='upper left',borderaxespad=0) # Place legend outside the axes, to the right
# Reserve room for the legend WITHIN the fixed 7.2 x 1.53 canvas
# instead of letting tight_layout() or the legend itself resize the figure
fig.subplots_adjust(left=0.09, right=0.78, top=0.82, bottom=0.28)
plt.savefig(f"{BASE_PATH}/figures/final/attn_kde.svg", dpi=300)
plt.show()

###### tdp43 
df_altair_input = pd.read_csv(f'{BASE_PATH}/data/figure_data/si_examples.csv')
df_altair_input = df_altair_input[df_altair_input['pid'] == 'TDP-43']

regions = pd.DataFrame({'region': ['NTD', 'RRM1', 'RRM2', 'CTD'],'start': [1, 104, 191, 274],'end': [103, 200, 262, 413],})
regions['label_pos'] = (regions['start'] + regions['end']) / 2
line_color_map = {'Attention Received': 'steelblue','Disorder': 'orange'}
fig, ax = plt.subplots(figsize=(panel_size(cols=4, rows=1)))
for i, row in regions.iterrows():
    ax.axvspan(row['start'], row['end'],color=colors[i % len(colors)],alpha=0.2,label=row['region'])
for var in df_altair_input['variable'].unique():
    sub = df_altair_input[df_altair_input['variable'] == var]
    ax.plot(sub['pos'], sub['value'], label=var, linewidth=1.2,color=line_color_map[var])
y_top = ax.get_ylim()[1]
for _, row in regions.iterrows():
    ax.text(row['label_pos'], y_top * 1.02, row['region'], ha='center', va='bottom', fontsize=5)
ax.set_xlabel("Position")
ax.set_ylabel("Value")
ax.set_ylim(top=y_top * 1.15)
ax.legend(loc='upper left',bbox_to_anchor=(1.02, 1),frameon=False)

fig.subplots_adjust(left=0.09, right=0.80, top=0.92, bottom=0.28)
plt.savefig(f"{BASE_PATH}/figures/final/tdp43.svg", dpi=300)
plt.show()

#########################################
# Figure 2 - AA Attention Allocation 
#########################################

###### mean AA attention rollout received plot
mean_aa_attn = pd.read_csv(f'{BASE_PATH}/data/figure_data/mean_aa_attn.csv')

amino_acids = mean_aa_attn["amino_acid"].unique()
x = np.arange(len(amino_acids))
fig, ax = plt.subplots(figsize=(7.2, 3.25), dpi=300)
for i, g in enumerate(['PDB', 'DisProt']):
    sub = mean_aa_attn[mean_aa_attn["Group"] == g]
    sub = sub.set_index("amino_acid").reindex(amino_acids)
    offsets = x + (i - 1) * 0.6
    ax.bar(offsets, sub["mean"], width=0.4, label=g, yerr=sub["se"], capsize=2, linewidth=0.8, color=group_color_map[g])
ax.set_xticks(x)
ax.set_xticklabels(amino_acids, rotation=0)
ax.set_xlabel("Amino Acid")
ax.set_ylabel("Mean Attention Rollout Received")
ax.legend(title="")
plt.tight_layout()
plt.savefig(f"{BASE_PATH}/figures/final/mean_aa_attn_received.svg", dpi=300)
plt.show()

####### Asp and Cys Heatmaps 
attn_occur_norm = pd.read_csv(f'{BASE_PATH}/data/figure_data/attn_occur_norm.csv')

mats = {}
for aa in ["C", "D"]:
    mats[aa] = {}
    for g in ['PDB', 'DisProt']:
        sub = attn_occur_norm[(attn_occur_norm["aa"] == aa) & (attn_occur_norm["Group"] == g)]
        mat = sub.pivot(index="head", columns="layer", values="attn_occur_norm")
        mats[aa][g] = mat

vmin = np.nanmin(attn_occur_norm["attn_occur_norm"])
vmax = np.nanmax(attn_occur_norm["attn_occur_norm"])
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(7.2, 3.25),constrained_layout=True, dpi =300)
im = None
for i, aa in enumerate(["C", "D"]):
    for j, g in enumerate(groups):
        if aa == "C":
            label = "Cys"
        elif aa == "D":
            label = "Asp"
        else:
            label = aa

        ax = axes[i, j]
        mat = mats[aa][g]
        im = ax.imshow(mat.values,aspect="equal",origin="lower",vmin=vmin,vmax=vmax, cmap="Blues")
        ax.set_title(f"{label} - {g}")
        if i == 1:
            ax.set_xlabel("Layer")
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel("Head")
        else:
            ax.set_yticklabels([])
        ax.set_xticks(np.arange(len(mat.columns)))
        xlabels = [str(c) if idx % 2 == 0 else "" for idx, c in enumerate(mat.columns)]
        ax.set_xticklabels(xlabels, rotation=90)

        ax.set_yticks(np.arange(len(mat.index)))
        ylabels = [str(h) if idx % 2 == 0 else "" for idx, h in enumerate(mat.index)]
        ax.set_yticklabels(ylabels)
cbar = fig.colorbar(im, ax=axes, shrink=0.8)
cbar.set_label("Average Attention Rollout", rotation=270, labelpad=12)
plt.savefig(f"{BASE_PATH}/figures/final/asp_cys_heatmaps.svg", dpi=300)
plt.show()



#########################
# Figure 3 - Pathogenicity
#########################
# import data
comb_data = pd.read_csv(f'{BASE_PATH}/data/figure_data/comb_data.csv')
log_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/log_df.csv')
ols_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/ols_df.csv')

# plot
# attention histograms
w, h = panel_size(cols=4, rows=1)  # (7.2, 1.53) — matches your other 4x1 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h))

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
plt.savefig(f'{BASE_PATH}/figures/final/clinvar_expanded_attention_percentile_distr.svg', dpi=300)
plt.show()

# attention by disorder percentile 
fig, ax = plt.subplots(figsize=(3.5, 1.53))
bins = sorted(comb_data['idr_perc_bin'].unique())
x = np.arange(len(bins))
width = 0.35
colors = {'Non-Disease-Relevant': 'steelblue', 'Disease-Relevant': 'orange'}
for i, (label, offset) in enumerate(zip(['Non-Disease-Relevant', 'Disease-Relevant'], [-width/2, width/2])):
    data_by_bin = [
        comb_data[(comb_data['idr_perc_bin'] == b) & (comb_data['mut_label_desc'] == label)]['attn_perc'].dropna()
        for b in bins
    ]
    bp = ax.boxplot(
        data_by_bin,
        positions=x + offset,
        widths=width * 0.8,
        patch_artist=True,
        manage_ticks=False,
        boxprops=dict(
            facecolor=colors[label],
            alpha=0.7,
            linewidth=0.4
        ),
        medianprops=dict(
            color='black',
            linewidth=0.6
        ),
        whiskerprops=dict(
            linewidth=0.4
        ),
        capprops=dict(
            linewidth=0.4
        ),
        flierprops=dict(
            marker='o',
            markersize=2,
            alpha=0.3
        )
    )
ax.set_xticks(x)
ax.set_xticklabels(bins)
ax.set_xlabel('Disorder Percentile Bins')
ax.set_ylabel('Attention Percentile')
ax.set_ylim(0, 1)
ax.set_title('Attention Percentile across Degrees of Disorder')

# wrap each label onto two lines so the legend box is narrower
legend_labels = ['Non-Disease-\nRelevant', 'Disease-\nRelevant']
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[l], alpha=0.7) for l in ['Non-Disease-Relevant', 'Disease-Relevant']]
ax.legend(handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
          fontsize=5, labelspacing=1.0, handlelength=1.2, handletextpad=0.5)

fig.subplots_adjust(left=0.16, right=0.76, top=0.72, bottom=0.28)
plt.savefig(f'{BASE_PATH}/figures/final/clinvar_expanded_attention_percentile_by_idr_percentile_bins.svg', dpi=300)
plt.show()

# disease aa counts 
mut_aa_counts = comb_data[comb_data['path_mut']==True].groupby('AA').size().reset_index()
mut_aa_counts.columns = ['AA', 'count']
mut_aa_counts = mut_aa_counts.sort_values('count', ascending=False)
fig, ax = plt.subplots(figsize=(3.5, 1.53))
ax.bar(mut_aa_counts['AA'], mut_aa_counts['count'], color = 'steelblue')
ax.set_xlabel('Amino Acid Type')
ax.set_ylabel('Count')
ax.set_title('Amino Acid Counts of\nDisease-Relevant WT Residues')
plt.tight_layout()
plt.savefig(f'{BASE_PATH}/figures/final/clinvar_expanded_disease_aa_count.svg', bbox_inches='tight')
plt.show()

# attention distribution by aa identity 

aa_groups = {
    **{aa: "Hydrophobic" for aa in ['A', 'I', 'L', 'M', 'F', 'V']},
    **{aa: "Polar"       for aa in ['S', 'Q', 'N', 'G', 'C', 'T', 'P']},
    **{aa: "Cation"     for aa in ['K', 'R', 'H']},
    **{aa: "Anion"      for aa in ['D', 'E']},
    **{aa: "Aromatic"   for aa in ['W', 'Y', 'F']},
}

comb_data["AA_group"] = comb_data["AA"].map(aa_groups)
group_order = ['Hydrophobic', 'Polar', 'Cation', 'Anion', 'Aromatic']
colors = {'Non-Disease-Relevant': 'steelblue', 'Disease-Relevant': 'orange'}
labels = ['Non-Disease-Relevant', 'Disease-Relevant']
width = 0.35
group_sizes = [len(set(aa for aa, g in aa_groups.items() if g == grp)) for grp in group_order]

print(comb_data[comb_data['AA_group'].isna()])
w, h = panel_size(cols=4, rows=1)  # (7.2, 1.53)
fig, axes = plt.subplots(1, len(group_order), figsize=(w, h), sharey=True,
                          gridspec_kw={'width_ratios': group_sizes, 'wspace': 0.12})

for i, g in enumerate(group_order):
    ax = axes[i]
    data = comb_data[comb_data['AA_group'] == g]
    aas = sorted(data.groupby('AA')['attn_perc'].median().sort_values(ascending=False).index)
    x = np.arange(len(aas))

    for j, (label, offset) in enumerate(zip(labels, [-width/2, width/2])):
        data_by_aa = [
            data[(data['AA'] == aa) & (data['mut_label_desc'] == label)]['attn_perc'].dropna()
            for aa in aas
        ]
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

# suptitle now sits INSIDE the canvas (y < 1) instead of floating above it
fig.suptitle('Attention Percentile Distributions by Amino Acid Group', fontsize=6, y=0.99)
fig.subplots_adjust(left=0.06, right=0.90, top=0.66, bottom=0.20)

plt.savefig(f'{BASE_PATH}/figures/final/clinvar_expanded_attn_percentile_by_aa_group.svg', dpi=300)
# NOT: bbox_inches='tight' — that's what let the canvas grow past 7.2 x 1.53
plt.show()


# log regression
log_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/log_df.csv')
log_df_melted = log_df[['idr_bin', 'rollout_z coef (adjusted)', 'conservation_z coef (adjusted)', 'LR_test']].rename(columns = {'rollout_z coef (adjusted)': 'Attention Rollout (z)', 'conservation_z coef (adjusted)' : 'Conservation (z)'})
log_df_melted = log_df_melted.melt(id_vars = 'idr_bin')
data = log_df_melted[log_df_melted['variable'] != 'LR_test']
idr_bins = sorted(data['idr_bin'].unique())
variables = data['variable'].unique()
n_bins = len(idr_bins)
x = np.arange(len(variables))
width = 0.6 / len(variables)
#colors = plt.cm.tab10(np.linspace(0, 1, len(variables)))
color_map = dict(zip(variables, ['orange', 'steelblue']))
w, h = panel_size(cols=2, rows=1)  # (3.5, 1.53)
fig, axes = plt.subplots(1, n_bins, figsize=(w, h), sharey=True)
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
plt.savefig(f'{BASE_PATH}/figures/final/clinvar_log_reg.svg', dpi=300)
plt.show()

# ols regression
ols_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/ols_df.csv')
ols_df_melted = ols_df[['beta_path_mut_adjusted', 'beta_path_mut_unadjusted', 'idr_bin']].melt(id_vars='idr_bin')
ols_df_melted['variable_label'] = ols_df_melted['variable'].str.extract(r'(unadjusted|adjusted)', flags=re.IGNORECASE)[0].str.capitalize()

idr_bins = sorted(ols_df_melted['idr_bin'].unique())
variable_labels = ols_df_melted['variable_label'].unique()

fig, axes = plt.subplots(1, n_bins, figsize=(w, h), sharey=True)

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
plt.savefig(f'{BASE_PATH}/figures/final/ols_clinvar.svg')
plt.show()

########################################
# Figure 4 - Structure/Contact Maps
########################################
# import data
combined_data = pd.read_csv(f'{BASE_PATH}/data/figure_data/combined_data.csv')

######### correlation distribution 
w, h = panel_size(cols=2, rows=2)  # (3.5, 3.25)
fig, ax = plt.subplots(figsize=(w, h), dpi=300)

type_color_map = {'Observed': 'steelblue', 'Mean Null (Permuted)': 'orange'}
types = combined_data['type'].unique()
bin_edges = np.histogram_bin_edges(combined_data['value'], bins=15)

for t in types:
    sub = combined_data[combined_data['type'] == t]
    ax.hist(sub['value'], bins=bin_edges, alpha=0.7, label=t,
            color=type_color_map[t])

ax.axvline(0, color='black', linestyle='--', linewidth=1.2, dashes=(5, 3))

ax.set_xlabel('Spearman Correlation')
ax.set_ylabel('Count')
ax.set_title('Observed vs Null Distribution (100 permutations each)', fontsize=7)
ax.legend(title='Distribution', fontsize=6, title_fontsize=6, frameon=False)

fig.subplots_adjust(left=0.20, right=0.95, top=0.83, bottom=0.22)  # left raised from 0.10
plt.savefig(f'{BASE_PATH}/figures/final/jac_md_observed_vs_null_distribution.svg', dpi=300)
plt.show()

##### agreement map 
agreement_color_map = {
    'Both Enriched': '#2166ac',
    'Both Depleted': '#d6604d',
    'Disagree': '#d3d3d3',
}
comparison = pd.read_csv(f'{BASE_PATH}/data/figure_data/comparison.csv')
pivot = comparison.pivot(index='res2', columns='res1', values='Agreement')
res1_order = sorted(comparison['res1'].unique())
res2_order = sorted(comparison['res2'].unique())
pivot = pivot.reindex(index=res2_order, columns=res1_order)

categories = list(agreement_color_map.keys())
cat_to_code = {c: i for i, c in enumerate(categories)}
code_grid = pivot.replace(cat_to_code).values.astype(float)

from matplotlib.colors import ListedColormap
cmap = ListedColormap([agreement_color_map[c] for c in categories])

w, h = panel_size(cols=2, rows=2)  # (3.5, 3.25)
fig, ax = plt.subplots(figsize=(w, h), dpi=300)
im = ax.imshow(code_grid, cmap=cmap, vmin=-0.5, vmax=len(categories) - 0.5, aspect='auto')

ax.set_xticks(range(len(res1_order)))
ax.set_xticklabels(res1_order, rotation=0, fontsize=5)
ax.set_yticks(range(len(res2_order)))
ax.set_yticklabels(res2_order, fontsize=5)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('MD vs ESM-2 Top Decile\nEnrichment Agreement', fontsize=6.5)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=agreement_color_map[c]) for c in categories]
fig.legend(handles, categories, loc='center left', bbox_to_anchor=(0.70, 0.5),
           frameon=False, fontsize=5.5, borderaxespad=0)

fig.subplots_adjust(left=0.16, right=0.68, top=0.82, bottom=0.14)
plt.savefig(f'{BASE_PATH}/figures/final/md_esm_top_decile_agreement.svg', dpi=300)
plt.show()

#########################
# Figure 5 - Rg MLP
#########################

# import data
df_eval = pd.read_csv(f'{BASE_PATH}/data/figure_data/df_eval.csv')
saxs_proteins = pd.read_csv(f'{BASE_PATH}/data/figure_data/saxs_proteins.csv')

# plot

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


r_idrome, _ = pearsonr(df_eval['Rg_true_norm'], df_eval['Rg_pred_norm'])
r_s_idrome, _ = spearmanr(df_eval['Rg_true_norm'], df_eval['Rg_pred_norm'])
r_flory_saxs, _       = pearsonr(saxs_proteins['saxs_norm'], saxs_proteins['Rg_pred_flory_norm'])
r_s_flory_saxs, _       = spearmanr(saxs_proteins['saxs_norm'], saxs_proteins['Rg_pred_flory_norm'])
r_starling, _ = pearsonr(saxs_proteins['saxs_norm'], saxs_proteins['starling_cuda_norm'])
r_s_starling, _ = spearmanr(saxs_proteins['saxs_norm'], saxs_proteins['starling_cuda_norm'])

# --- build the combined 4x1 figure ---
w, h = panel_size(cols=4, rows=1)  # (7.2, 1.53)
fig, axes = plt.subplots(1, 3, figsize=(w, h), dpi=300)

plot_comparison_color(
    axes[0], x=df_eval['Rg_true_norm'], y=df_eval['Rg_pred_norm'],
    xlabel='Tesei Rg\nFlory Normalized (Å)', ylabel='MLP Predicted Rg\nFlory Normalized (Å)',
    title=['IDRome-MLP', f'Pearson r = {r_idrome:.3f}, Spearman r = {r_s_idrome:.3f}'],
    alpha=1, s=8
)

plot_comparison_color(
    axes[1], x=saxs_proteins['saxs_norm'], y=saxs_proteins['Rg_pred_flory_norm'],
    xlabel='Experimental SAXS Rg\nFlory Normalized (Å)', ylabel='Predicted Rg\nFlory Normalized (Å)',
    title=['SAXS-MLP', f'Pearson r = {r_flory_saxs:.3f}, Spearman r = {r_s_flory_saxs:.3f}'],
    alpha=1, s=8
)

plot_comparison_color(
    axes[2], x=saxs_proteins['saxs_norm'], y=saxs_proteins['starling_cuda_norm'],
    xlabel='Experimental SAXS Rg\nFlory Normalized (Å)', ylabel='STARLING Cuda Rg\nFlory Normalized (Å)',
    title=['SAXS-STARLING', f'Pearson r = {r_starling:.3f}, Spearman r = {r_s_starling:.3f}'],
    alpha=1, s=8
)

fig.subplots_adjust(
    left=0.055,
    right=0.99,
    top=0.82,
    bottom=0.23,
    wspace=0.22
)
plt.savefig(f'{BASE_PATH}/figures/final/rg_mlp.svg', dpi=300)
plt.show()


#########################
# Figure 6 - ES Ablation 
########################## 

######## sweep df 

N_RANDOM_RUNS=25
sweep_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/es_ablation_sweep_df_idrome_random_consecutive.csv')
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
std_df = sweep_df[['n_heads', 'delta_spearman_random_se']].rename(
    columns={'delta_spearman_random_se': 'se'})
sweep_long = sweep_long.merge(std_df, on='n_heads', how='left')
sweep_long['se']         = sweep_long['se'].where(sweep_long['ablation'] == 'Random', 0)
sweep_long['delta_upper'] = sweep_long['delta'] + sweep_long['se']
sweep_long['delta_lower'] = sweep_long['delta'] - sweep_long['se']

color_scale = {
    'Negative (D/E)': '#E8300C',
    'Positive (K/R)': '#2367B0',
    'Random': '#999999',
}

w, h = panel_size(cols=2, rows=1)  # (3.5, 1.53)
fig, ax = plt.subplots(figsize=(w, h), dpi=300)

# --- band: shaded confidence region, Random ablation only ---
random_sub = sweep_long[sweep_long['ablation'] == 'Random'].sort_values('n_heads')
ax.fill_between(
    random_sub['n_heads'], random_sub['delta_lower'], random_sub['delta_upper'],
    color=color_scale['Random'], alpha=0.2, linewidth=0
)

# --- lines + points, one pass per ablation category ---
for ablation, color in color_scale.items():
    sub = sweep_long[sweep_long['ablation'] == ablation].sort_values('n_heads')
    ax.plot(sub['n_heads'], sub['delta'], color=color, linewidth=1.2, label=ablation)
    ax.scatter(sub['n_heads'], sub['delta'], color=color, s=8, zorder=3, edgecolors='none')

ax.set_xlabel('Number of Heads Ablated', fontsize=5.5)
ax.set_ylabel('Δ(Ablated − Baseline) \n Corr(Pred & True Rg) ',
               fontsize=5.5, linespacing=1.3)
ax.set_title(f'Head Ablation Sweep', fontsize=6.5)
ax.tick_params(labelsize=4.5)

ax.legend(title='Ablation', fontsize=5, title_fontsize=5.5, frameon=False,
          bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

fig.subplots_adjust(left=0.16, right=0.72, top=0.78, bottom=0.32)
plt.savefig(f'{BASE_PATH}/figures/final/head_ablation_sweep_.svg', dpi=300)
plt.show()

##### scatter plots 

results_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/results_df.csv')
saxs_results_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/saxs_results_df.csv')

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

blue_scale_cmap = LinearSegmentedColormap.from_list('white_blue', ['white', '#2367B0'])

def plot_comparison_color(x, y, xlabel, ylabel, title, save_path=None, alpha=0.3, s=20,
                           color=None, color_title='', figsize=(1.65, 1.53), dpi=300):
    x = np.asarray(x); y = np.asarray(y); color = np.asarray(color)
    abs_max = np.abs(color).max()
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='6%', pad=0.05)   # tighter than before

    sc = ax.scatter(x, y, c=color, cmap=blue_scale_cmap, vmin=0, vmax=abs_max,
                      alpha=alpha, s=s, edgecolors='none')
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

label = 'idrome'
plot = plot_comparison_color(
    results_df['rg_baseline_scaled'], results_df['rg_ablated_scaled'],
    'Baseline Predicted Rg\n(Flory Normalized)', 'Ablated Predicted Rg\n(Flory Normalized)',
    f'{label.upper()} - Negative', save_path=f'{BASE_PATH}/figures/final/es_negative_{label}_scatter.svg',
    alpha=1, s=12, color=results_df['frac_DE'], color_title='Frac DE')

label = 'saxs'
plot = plot_comparison_color(
    saxs_results_df['rg_baseline_scaled'], saxs_results_df['rg_ablated_scaled'],
    'Baseline Predicted Rg\n(Flory Normalized)', 'Ablated Predicted Rg\n(Flory Normalized)',
    f'{label.upper()} - Negative', save_path=f'{BASE_PATH}/figures/final/es_negative_{label}_scatter.svg',
    alpha=1, s=12, color=saxs_results_df['frac_DE'], color_title='Frac DE')

####### kde 

from scipy.stats import gaussian_kde

results_df['bigger_or_smaller'] = results_df['rg_ablated_scaled'] - results_df['rg_baseline_scaled']
results_df['frac_DE_quartile'] = pd.qcut(results_df['frac_DE'], q=10,
                                          labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])

deciles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
blue_colors = ['#dbeafe', '#bfdbfe', '#93c5fd', '#60a5fa', '#3b82f6',
               '#2563eb', '#1d4ed8', '#1e40af', '#1e3a8a', '#172554']
decile_color_map = dict(zip(deciles, blue_colors))

def plot_kde_by_decile(ax, data, field, xlabel, color_field='frac_DE_quartile'):
    for q in deciles:
        vals = data.loc[data[color_field] == q, field].dropna().values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, kde(xs), color=decile_color_map[q], linewidth=0.8, label=q)
    ax.set_xlabel(xlabel, fontsize=4.2, labelpad=2)
    ax.set_ylabel('Density', fontsize=4.2, labelpad=2)
    ax.tick_params(labelsize=3.2, pad=1)

w, h = panel_size(cols=2, rows=1)  # (1.65, 3.25)
fig, axes = plt.subplots(2, 1, figsize=(w, h), dpi=300)

plot_kde_by_decile(axes[0], results_df, 'bigger_or_smaller', 'Ablated Pred Rg -\nBaseline Pred Rg')
axes[0].axvline(0, color='red', linewidth=1.0)

plot_kde_by_decile(axes[1], results_df, 'rg_baseline_scaled', 'Rg Baseline')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Frac DE\nDecile', fontsize=3.0, title_fontsize=3.2,
           loc='center left', bbox_to_anchor=(0.83, 0.5), frameon=False, handlelength=0.8,
           handletextpad=0.3, labelspacing=0.25, borderaxespad=0)

fig.suptitle('Negative Head Ablation - IDRome', fontsize=4.6, y=0.99)
fig.subplots_adjust(left=0.20, right=0.82, top=0.94, bottom=0.10, hspace=0.42)

plt.savefig(f'{BASE_PATH}/figures/final/es_ablation_kde.svg', dpi=300)
plt.show()

#########################
# Figure A1 
#########################

protein_sets = pd.read_csv(f'{BASE_PATH}/data/protein_input_sets.csv')
protein_sets['Group'] = np.where(
    protein_sets['ProteinID'].str.contains('pdb', case=False, na=False), 'pdb',
    np.where(protein_sets['ProteinID'].str.contains('disprot', case=False, na=False), 'disprot', None)
)
protein_lengths = protein_sets.copy()
protein_lengths['seq_len'] = protein_lengths['Sequence'].apply(len)
protein_lengths['Group'] = protein_lengths['Group'].replace({'pdb': 'PDB', 'disprot': 'DisProt'})

GROUP_COLORS = {'PDB': 'orange', 'DisProt': 'steelblue'}

groups = ['DisProt', 'PDB']
bins = np.histogram_bin_edges(protein_lengths['seq_len'], bins=40)  # shared edges so bars align
fig, ax = plt.subplots(figsize=panel_size(cols=2, rows=1), layout='constrained')
for g in groups:
    ax.hist(protein_lengths.loc[protein_lengths.Group == g, 'seq_len'],
            bins=bins, label=g, color=GROUP_COLORS[g], alpha=0.55, edgecolor='none')
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Count")
ax.set_title("PDB & DisProt Sequence Length")
ax.legend(title="Group", loc='upper right')
plt.savefig(f"{BASE_PATH}/figures/final/input_sets_seqlen_distribution.svg", dpi=300)
plt.show()


idr_df = pd.read_csv(f'{BASE_PATH}/data/figure_data/idr_df.csv')

GROUP_COLORS = {'PDB': 'orange', 'DisProt': 'steelblue'}
groups = ['DisProt', 'PDB']

w, h = panel_size(cols=2, rows=1)  # (3.5, 1.53)
fig, ax = plt.subplots(figsize=(w, h), dpi=300, layout='constrained')
fig.get_layout_engine().set(w_pad=0.15, h_pad=0.15)  # extra buffer so labels don't sit right on the edge

for g in groups:
    vals = idr_df.loc[idr_df['Group'] == g, 'idr'].dropna().values
    kde = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(xs, kde(xs), color=GROUP_COLORS[g], linewidth=1.2, label=g)

ax.set_xlabel('# of Disordered Residues per Sequence', fontsize=5.5)
ax.set_ylabel('Density', fontsize=5.5)
ax.tick_params(labelsize=4.5)
ax.legend(title='Group', fontsize=5, title_fontsize=5.5, frameon=False, loc='upper right')

plt.savefig(f"{BASE_PATH}/figures/final/input_sets_disorder.svg", dpi=300)
plt.show()


comb_props = pd.read_csv(f'{BASE_PATH}/data/figure_data/comb_props.csv')

GROUP_COLORS = {'PDB': 'orange', 'DisProt': 'steelblue'}
groups = ['DisProt', 'PDB']
amino_acids = sorted(comb_props['amino_acid'].unique())
x = np.arange(len(amino_acids))
width = 0.8 / len(groups)

w, h = panel_size(cols=4, rows=1)  # (7.2, 1.53)
fig, ax = plt.subplots(figsize=(w, h), dpi=300, layout='constrained')
fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1)

for i, g in enumerate(groups):
    sub = comb_props[comb_props['Group'] == g].set_index('amino_acid').reindex(amino_acids)
    offsets = x + (i - len(groups)/2) * width + width/2
    ax.bar(offsets, sub['occurrence'], width=width, label=g, color=GROUP_COLORS[g])

ax.set_xticks(x)
ax.set_xticklabels(amino_acids, fontsize=5)
ax.set_xlabel('Amino Acid', fontsize=6)
ax.set_ylabel('Proportional\nFrequency (%)', fontsize=6, linespacing=1.1)
ax.tick_params(axis='y', labelsize=5)
ax.legend(title='Group', fontsize=5.5, title_fontsize=6, frameon=False, loc='upper right')

plt.savefig(f"{BASE_PATH}/figures/final/aa_proportional_frequency.svg", dpi=300)
plt.show()
print('saved aa proportional frequency plot')



#########################
# Figure A2
#########################
si_examples = pd.read_csv(f'{BASE_PATH}/data/figure_data/si_examples.csv')

line_color_map = {'Attention Received': 'steelblue', 'Disorder': 'orange'}
pids = si_examples['pid'].unique()
n = len(pids)

# Keep the same overall figure footprint as a 4x4 grid,
# but lay out pids as n rows x 1 column
w, h = panel_size(cols=4, rows=4)  # (7.2, 7.2) — total canvas size unchanged
fig, axes = plt.subplots(n, 1, figsize=(w, h), dpi=300)
axes_flat = np.atleast_1d(axes)  # handles n=1 gracefully

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

plt.savefig(f"{BASE_PATH}/figures/final/si_examples.svg", dpi=300)
plt.show()

#################
# Figure A4
#################
training_curve_long = pd.read_csv(f'{BASE_PATH}/data/figure_data/training_curve_long.csv')
best_epoch = int(np.argmin(training_curve_long[training_curve_long['dataset']=='Validation'].mse_loss))

dataset_color_map = {'Training': 'steelblue', 'Validation': 'orange'}  # replace with your actual 'dataset' values

w, h = panel_size(cols=4, rows=2)  # (3.5, 6.69)
fig, ax = plt.subplots(figsize=(w, h), dpi=300)

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

plt.savefig(os.path.join(BASE_PATH, "figures/final/mlp_final_curves.svg"), dpi=300)
plt.show()



#################
# Figure A4
#################
attn_occur_norm = pd.read_csv(f'{BASE_PATH}/data/figure_data/attn_corr_norm_es.csv')

mean_pos = (attn_occur_norm[attn_occur_norm['aa'].isin(['R', 'K'])].groupby(['head', 'layer'])['attn_occur_norm'].mean().reset_index())
mean_neg = (attn_occur_norm[attn_occur_norm['aa'].isin(['D', 'E'])].groupby(['head', 'layer'])['attn_occur_norm'].mean().reset_index())

heads_pos_ranked = mean_pos.sort_values('attn_occur_norm', ascending=False).reset_index(drop=True)
heads_neg_ranked = mean_neg.sort_values('attn_occur_norm', ascending=False).reset_index(drop=True)

unique_aas = sorted(attn_occur_norm['aa'].unique())
n = len(unique_aas)
ncols = 2
nrows = int(np.ceil(n / ncols))

vmin = attn_occur_norm['attn_occur_norm'].min()
vmax = attn_occur_norm['attn_occur_norm'].max()

w, h = panel_size(cols=4, rows=2)  # (3.5, 6.69) — was cols=4, rows=4, which gave (7.2, 7.2)
fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=300)
axes_flat = axes.flatten()

im = None
for i, ax in enumerate(axes_flat):
    if i >= n:
        ax.axis('off')
        continue
    aa = unique_aas[i]
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
fig.subplots_adjust(left=0.12, right=0.85, top=0.93, bottom=0.06, hspace=0.62, wspace=0.20)  # wspace was 0.55

cax = fig.add_axes([0.88, 0.15, 0.03, 0.65])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=4)

plt.savefig(f"{BASE_PATH}/figures/final/DEKR_heatmaps.svg", dpi=300)
plt.show()