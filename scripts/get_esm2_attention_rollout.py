import torch
import pandas as pd
import numpy as np
import sys
import gzip
import pickle
import sys
import joblib
from tqdm import tqdm
from attention_functions import safe_load, attention_rollout

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 

print(torch.__version__)         
print(torch.version.cuda)          
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


'''
Script to compute attention rollout for Disordered and Folded proteins 

'''

# load the attention matrices for each protein and compute rollout 
model = 'original'
attn_rollout = dict()

for i in tqdm(range(0,22)): 

    # load esm2 attention matrices
    attn_i = safe_load(f'{BASE_PATH}/attention_matrices_chunks/attention_matrices_labeled_{i}_{model}.joblib')
    
    # for each protein in the batch compute attention rollout
    for protein in attn_i.keys(): 
        protein_attn = attn_i[protein]
        sequence = protein_attn['sequence']
        attentions = torch.tensor(protein_attn['attention_matrix'], device = device)
        raw_sequence_lengths = protein_attn['sequence_length']

        rollout = attention_rollout(attentions, raw_sequence_lengths, discard_ratio=0.9, head_fusion="mean", device = device)

        attn_rollout[protein] = {'sequence' : sequence, 'rollout' : rollout}
        del rollout, protein_attn, attentions
     
    del attn_i


joblib.dump(attn_rollout, f"{BASE_PATH}/attention_matrices_chunks/attention_rollout_comb_{model}.joblib", compress=3)
