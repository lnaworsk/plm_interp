'''
Build the public copy of the figure source data used by scripts/figures.py.

Reads the full analysis outputs in data/figure_data/ (gitignored; written by the analysis
scripts) and writes data/figure_data_public/ (committed), which:
  - keeps only the 16 files figures.py needs
  - removes third-party reference values that are not ours to redistribute:
      * saxs_proteins.csv: experimental SAXS Rg and STARLING predictions (Novak et al., bioRxiv 2025)
      * df_eval.csv: MD-derived Rg from IDRome (Tesei et al., Nature 2024; GPL-3.0)
    figures.py merges these back in from their original sources.
  - drops the repeated full-length sequence column from comb_data.csv (size only), replacing it
    with protein_idx = order of first appearance of each unique Seq in data/clinvar_sequences.csv
  - gzips the two large files (comb_data, disorder_comp_concat); pandas reads .csv.gz directly

Run after regenerating data/figure_data/:  python scripts/make_public_figure_data.py
'''

import shutil
from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).resolve().parent.parent
SRC = BASE_PATH / 'data' / 'figure_data'
OUT = BASE_PATH / 'data' / 'figure_data_public'
OUT.mkdir(parents=True, exist_ok=True)

# files copied unchanged (all our own computed results)
copy_as_is = [
    'umap_df.csv', 'si_examples.csv', 'mean_aa_attn.csv', 'attn_occur_norm.csv',
    'log_df.csv', 'ols_df.csv', 'combined_data.csv', 'comparison.csv',
    'es_ablation_sweep_df_idrome_random_consecutive.csv', 'idr_df.csv',
    'comb_props.csv', 'training_curve_long.csv',
]
for f in copy_as_is:
    shutil.copy(SRC / f, OUT / f)

# large file: gzip only
pd.read_csv(SRC / 'disorder_comp_concat.csv').to_csv(
    OUT / 'disorder_comp_concat.csv.gz', index=False, compression='gzip')

# remove third-party columns
pd.read_csv(SRC / 'saxs_proteins.csv', skipinitialspace=True)[
    ['sequence', 'seqlen', 'Rg_pred', 'Rg_pred_flory_norm']
].to_csv(OUT / 'saxs_proteins.csv', index=False)

pd.read_csv(SRC / 'df_eval.csv')[
    ['seq_name', 'N', 'Rg_pred', 'Rg_pred_norm']
].to_csv(OUT / 'df_eval.csv', index=False)

# drop repeated sequence column, keep a protein index; gzip
comb = pd.read_csv(SRC / 'comb_data.csv')
comb.insert(0, 'protein_idx', pd.factorize(comb['seq'])[0])
comb.drop(columns=['seq', 'labels']).to_csv(
    OUT / 'comb_data.csv.gz', index=False, compression='gzip')

for p in sorted(OUT.iterdir()):
    print(f'{p.stat().st_size / 1e6:8.1f} MB  {p.name}')
