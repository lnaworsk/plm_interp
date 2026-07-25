
import pandas as pd
import gzip
from Bio import SeqIO
from attention_functions import *
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 
'''
Script to define Disprot and PDB datasets for ESM2 attention analysis

DisProt Fasta File: DisProt release_2025_06 with_ambiguous_evidences
PDB: https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
'''

disprot = read_fasta(f'{BASE_PATH}/data/disprot')
disprot = disprot.drop_duplicates(subset = 'Sequence')
disprot['seq_len'] = disprot['Sequence'].apply(len)
disprot = disprot[disprot['seq_len']<1024] 
disprot = disprot[disprot['seq_len']>50] 
disprot['Group'] = 'Disprot'

# get ordered data (PDB)
file_path = f"{BASE_PATH}/data/pdb_seqres.txt.gz"

with gzip.open(file_path, 'rt') as f:
    records = list(SeqIO.parse(f, 'fasta'))

pdb = pd.DataFrame({'sequence':[str(r.seq) for r in records]})
pdb = pdb[~pdb['sequence'].str.contains('X|U|O')]
pdb = pdb.drop_duplicates(subset = 'sequence')
pdb['seq_len'] = pdb['sequence'].apply(len)
pdb = pdb[pdb['seq_len']<1024] 
pdb = pdb[pdb['seq_len']>50] 
pdb['Group'] = 'PDB'

# select pdb to mirror the seq len distribution label of the disordered proteins 

# get the number of disordered proteins in each protein length bin 
disordered_lengths = disprot['seq_len']
bins = np.linspace(disordered_lengths.min(), disordered_lengths.max(), 21)  # 20 bins
disordered_bins = np.digitize(disordered_lengths, bins)
bin_counts = pd.Series(disordered_bins).value_counts().sort_index()

# assign the ordered proteins to the same length bins 
ordered_bins = np.digitize(pdb['seq_len'], bins)
pdb = pdb.assign(length_bin=ordered_bins)

# subsample the ordered to match the length distribution of the ordered proteins 
subsampled_ordered = []
for bin_idx, count in bin_counts.items():
    candidates = pdb[pdb['length_bin'] == bin_idx]
    if len(candidates) >= count:
        sampled = candidates.sample(n=count, random_state=42)
    else:
        sampled = candidates  # take all if fewer than needed
    subsampled_ordered.append(sampled)
# Combine all sampled subsets
subsampled_pdb = pd.concat(subsampled_ordered).reset_index(drop=True)

disprot['ProteinID'] = ['disprot_protein' + str(i + 1) for i in range(len(disprot))]
subsampled_pdb['ProteinID'] = ['pdb_protein' + str(i + 1) for i in range(len(subsampled_pdb))]
input_data = pd.concat([disprot[['Sequence', 'ProteinID']], subsampled_pdb.rename(columns = {'sequence' : 'Sequence'})[['Sequence', 'ProteinID']]])
input_data.to_csv(f'{BASE_PATH}/data/protein_input_sets.csv', index = False)
print(f'Saved input protein sets to {BASE_PATH}/data/protein_input_sets.csv')