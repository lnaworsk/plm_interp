import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
from joblib import Parallel, delayed
import os 
import torch 
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 

'''
Script to analyze IDP MD-derived and ESM-2-derived contact maps 
'''


def compute_null_distribution_for_protein(seq_name, df_seq, n_permutations = 1000, seed = 42):

    contact_strengths = df_seq["ContactStrength"].values
    frac_values = df_seq["Frac"].values
    
    # Observed correlation
    observed_corr = spearmanr(contact_strengths, frac_values)[0]
    
    # Vectorized permutations 
    n_contacts = len(contact_strengths)
    
    # Create matrix of permuted indices
    rng = np.random.default_rng(seed)

    perm_indices = np.array([rng.permutation(n_contacts) for _ in range(n_permutations)])

    # Apply all permutations at once
    shuffled_matrix = contact_strengths[perm_indices]
    
    # Compute correlations for all permutations 
    null_corrs = np.array([spearmanr(shuffled_matrix[i], frac_values)[0] for i in range(n_permutations)])
    
    # Calculate statistics
    null_mean = np.mean(null_corrs)
    null_std = np.std(null_corrs)
    z_score = (observed_corr - null_mean) / null_std if null_std > 0 else np.nan
    p_value = (np.sum(null_corrs >= observed_corr) + 1) / (n_permutations + 1)
    
    return {
        'seq_name': seq_name,
        'observed_r': observed_corr,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score,
        'p_value': p_value,
        'null_corrs': null_corrs  # Return for aggregate distribution
    }


# -------------------------------------------
#     Correlation of MD & Jac Maps with Null 
# -------------------------------------------
variable_sigma = True

### Load upper triangle data
df_upper1 = pd.read_csv(os.path.join(BASE_PATH, 'data/jacobian/jacobian_md_frac_batch1_upper.csv'))
df_upper2 = pd.read_csv(os.path.join(BASE_PATH, 'data/jacobian/jacobian_md_frac_batch2_upper.csv'))
df_upper3 = pd.read_csv(os.path.join(BASE_PATH, 'data/jacobian/jacobian_md_frac_batch3_upper.csv'))
df_upper4 = pd.read_csv(os.path.join(BASE_PATH, 'data/jacobian/jacobian_md_frac_batch4_upper.csv'))
df_upper5 = pd.read_csv(os.path.join(BASE_PATH, 'data/jacobian/jacobian_md_frac_batch5_upper.csv'))
df_upper6 = pd.read_csv(os.path.join(BASE_PATH, 'data/jacobian/jacobian_md_frac_batch6_upper.csv'))
df_upper7 = pd.read_csv(os.path.join(BASE_PATH, 'data/jacobian/jacobian_md_frac_batch7_upper.csv'))


df_upper_all = pd.concat([df_upper1, df_upper2, df_upper3, df_upper4, 
                            df_upper5, df_upper6, df_upper7]).drop_duplicates()

if variable_sigma == True: 
    orig_len = len(df_upper_all)
    md_variable = pd.concat([pd.read_csv(os.path.join(BASE_PATH, f"data/jacobian/md_variable_sigma_batch{i}.csv")) for i in range(1, 8)],ignore_index=True)
    df_upper_all = df_upper_all.drop(columns = 'Frac').merge(md_variable, on = ['Residue_i', 'Residue_j', 'seq_name']).drop_duplicates()
    merged_len = len(df_upper_all)

    if orig_len == merged_len: 
        print('No loss')
    else: 
        print(f"Original : {orig_len} | Merged : {merged_len}")

grouped = df_upper_all.groupby('seq_name')
seq_names = list(grouped.groups.keys())
seq_names_filtered = [name for name in seq_names if len(grouped.get_group(name)) >= 10]
print(f"Processing {len(seq_names_filtered)} proteins with >=10 contacts...")

print("Computing null distributions (100 permutations per protein)...")
results = Parallel(n_jobs=8)(
    delayed(compute_null_distribution_for_protein)(
        name, grouped.get_group(name), n_permutations=100, seed=42+i
    ) for i, name in enumerate(seq_names_filtered)
)

null_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'null_corrs'} for r in results])
all_null_corrs = np.concatenate([r['null_corrs'] for r in results])

# -------------------------------------
#         Statistical Summary
# -------------------------------------

print("\n" + "="*60)
print("NULL MODEL ANALYSIS RESULTS")
print("="*60)

# Observed distribution stats
print("\nOBSERVED CORRELATIONS:")
print(f"  Median r = {np.median(null_df['observed_r']):.4f}")
print(f"  Q1-Q3: [{np.percentile(null_df['observed_r'], 25):.4f}, "
        f"{np.percentile(null_df['observed_r'], 75):.4f}]")
print(f"  Mean r = {np.mean(null_df['observed_r']):.4f} ± {np.std(null_df['observed_r']):.4f}")

# Null distribution stats
print("\nNULL EXPECTATION:")
print(f"  Mean of mean null r = {np.mean(null_df['null_mean']):.4f}")
print(f"  Overall Mean null r = {np.mean(all_null_corrs):.4f} ± {np.std(all_null_corrs):.4f}")

# Statistical significance
n_significant = np.sum(null_df['p_value'] < 0.05)
n_total = len(null_df)
print(f"\nSTATISTICAL SIGNIFICANCE:")
print(f"  Proteins with p < 0.05: {n_significant}/{n_total} ({100*n_significant/n_total:.1f}%)")
print(f"  Median z-score: {np.median(null_df['z_score']):.2f}")

# Effect size
mean_observed = np.mean(null_df['observed_r'])
mean_null = np.mean(all_null_corrs)
effect_size = mean_observed - mean_null
print(f"\nEFFECT SIZE:")
print(f"  Difference from null: {effect_size:.4f}")
print(f"  R² (variance explained): ~{mean_observed**2:.3f} ({100*mean_observed**2:.1f}%)")

print("="*60 + "\n")

# ----------------------------------------------
#     Visualization of Correlation Distribution 
# ----------------------------------------------

observed_data = pd.DataFrame({'value': null_df['observed_r'], 'type': 'Observed'})
null_data = pd.DataFrame({'value': null_df['null_mean'], 'type': 'Mean Null (Permuted)'})
combined_data = pd.concat([observed_data, null_data], ignore_index=True)
combined_data.to_csv(f'{BASE_PATH}/data/figure_data/combined_data.csv', index = False)

fig, ax = plt.subplots(dpi=300)
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
plt.savefig(f'{BASE_PATH}/figures/jac_md_observed_vs_null_distribution.svg', dpi=300)
plt.show()

# -------------------------
#     Top Decile Analysis  
# -------------------------

supp_table = pd.read_excel(f'{BASE_PATH}/data/Supplementary_Table_3.xlsx')
df_comb_upper = df_upper_all.merge(supp_table[['seq_name', 'sequence']], on = 'seq_name', how = 'left')
df_comb_upper['Residue_i_AA'] = df_comb_upper.apply(lambda x: x['sequence'][x['Residue_i']], axis=1)
df_comb_upper['Residue_j_AA'] = df_comb_upper.apply(lambda x: x['sequence'][x['Residue_j']], axis=1)
df_comb_upper['Residue Pair'] = df_comb_upper['Residue_i_AA'] + '-' + df_comb_upper['Residue_j_AA']
df_comb_upper['Residue Pair'] = (df_comb_upper['Residue Pair'].str.split('-').apply(lambda x: '-'.join(sorted(x))))
total_count_df = df_comb_upper.groupby('Residue Pair')['seq_name'].count().reset_index().rename(columns = {'seq_name' : 'total_count'})

cutoff_jac = df_comb_upper['ContactStrength'].quantile(0.9)
cutoff_md = df_comb_upper['Frac'].quantile(0.9)
top_10_percent_jac = df_comb_upper[df_comb_upper['ContactStrength'] >= cutoff_jac].sort_values(by='ContactStrength', ascending=False)
top_10_percent_md = df_comb_upper[df_comb_upper['Frac'] >= cutoff_md].sort_values(by='Frac', ascending=False)

# ensure both dataframes have the same number of samples for comparison
num_rows = min(top_10_percent_jac.shape[0], top_10_percent_md.shape[0])
top_10_percent_jac = top_10_percent_jac.head(num_rows)
top_10_percent_md = top_10_percent_md.head(num_rows)

print('Done loading data')

############ Plot Top 10% Residue Pairs 

count_top_10_jac = top_10_percent_jac.groupby('Residue Pair').size().reset_index(name='counts')
count_top_10_jac = count_top_10_jac.merge(total_count_df, on='Residue Pair', how='left')
count_top_10_jac['scaled_counts'] = count_top_10_jac['counts'] / count_top_10_jac['total_count']

count_top_10_md = top_10_percent_md.groupby('Residue Pair').size().reset_index(name='counts')
count_top_10_md = count_top_10_md.merge(total_count_df, on='Residue Pair', how='left')
count_top_10_md['scaled_counts'] = count_top_10_md['counts'] / count_top_10_md['total_count']

############ Plot Top 10% Counts Heatmap ###################

count_top_10_jac['method'] = 'ESM-2 Jacobian'
count_top_10_md['method'] = 'MD Fraction in Contact'
comb_counts = (pd.concat([count_top_10_jac, count_top_10_md], ignore_index=True).rename(columns={'counts': 'Counts','scaled_counts': 'Scaled Counts'}))
comb_counts[['res1', 'res2']] = comb_counts['Residue Pair'].str.split('-', expand=True)

method_totals = (comb_counts.groupby('method')['Counts'].sum().rename('method_total').reset_index())
all_residue_pairs = total_count_df['total_count'].sum()

# Calculate enrichment/depletion

comb_counts['tot_prop_freq'] = (comb_counts['total_count'] / all_residue_pairs * 100)
wide = comb_counts.merge(method_totals, on='method')
wide['top_decile_prop_freq'] = (wide['Counts'] / wide['method_total'] * 100)
wide['delta'] = (wide['top_decile_prop_freq'] - wide['tot_prop_freq'])
wide['enriched'] = np.where(wide['delta'] > 0, 'Enriched', 'Depleted')
comparison = (wide.pivot_table(index=['res1', 'res2'],columns='method', values='delta').reset_index())
comparison.columns.name = None
comparison = comparison.rename(columns={'MD Fraction in Contact': 'delta_md','ESM-2 Jacobian': 'delta_esm'})
comparison['Agreement'] = np.select([(comparison['delta_md'] > 0) & (comparison['delta_esm'] > 0),(comparison['delta_md'] < 0) & (comparison['delta_esm'] < 0)],['Both Enriched','Both Depleted'],default='Disagree')
comparison.to_csv(f'{BASE_PATH}/data/figure_data/comparison.csv',index=False)


# plotting agreement map 

agreement_color_map = {'Both Enriched': '#2166ac','Both Depleted': '#d6604d','Disagree': '#d3d3d3',}
pivot = comparison.pivot(index='res2', columns='res1', values='Agreement')
res1_order = sorted(comparison['res1'].unique())
res2_order = sorted(comparison['res2'].unique())
pivot = pivot.reindex(index=res2_order, columns=res1_order)

categories = list(agreement_color_map.keys())
cat_to_code = {c: i for i, c in enumerate(categories)}
code_grid = pivot.replace(cat_to_code).values.astype(float)
cmap = ListedColormap([agreement_color_map[c] for c in categories])

fig, ax = plt.subplots(dpi=300)
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
plt.savefig(f'{BASE_PATH}/figures/md_esm_top_decile_agreement.svg', dpi=300)
plt.show()

# plotting agreement map with magnitudes 

esm_pivot = comparison.pivot(index='res2',columns='res1',values='delta_esm')
md_pivot = comparison.pivot(index='res2',columns='res1',values='delta_md')
# Ensure same residue ordering
res1_order = sorted(comparison['res1'].unique())
res2_order = sorted(comparison['res2'].unique())
esm_pivot = esm_pivot.reindex(index=res2_order, columns=res1_order)
md_pivot = md_pivot.reindex(index=res2_order, columns=res1_order)
cmap = 'RdBu'   # red = negative, blue = positive


def plot_enrichment_heatmap(data, title, filename):
    fig, ax = plt.subplots(dpi=300)

    im = ax.imshow(data,cmap=cmap,aspect='auto')
    ax.set_xticks(range(len(res1_order)))
    ax.set_xticklabels(res1_order, fontsize=5)
    ax.set_yticks(range(len(res2_order)))
    ax.set_yticklabels(res2_order, fontsize=5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=7)

    cbar = fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cbar.set_label('Enrichment Δ (%)', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.subplots_adjust(left=0.16,right=0.88,top=0.85,bottom=0.14)

    plt.savefig(f'{BASE_PATH}/figures/{filename}.svg',dpi=300,bbox_inches='tight')
    plt.show()


plot_enrichment_heatmap(esm_pivot.values,'ESM-2 Jacobian Top Decile\nResidue Pair Enrichment','esm2_top_decile_enrichment')
plot_enrichment_heatmap(md_pivot.values,'MD Fraction in Contact Top Decile\nResidue Pair Enrichment','md_top_decile_enrichment')
