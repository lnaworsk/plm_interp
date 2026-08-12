import torch
import esm
import pandas as pd
import sys
import platform
import joblib
import gc
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 

'''
Script to get esm2 attention matrices 

'''

######## Load ESM-2 #######

print(torch.__version__)         
print(torch.version.cuda)          
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device) 
batch_converter = alphabet.get_batch_converter()
model.eval()

###### Read in DisProt and PDB Protein Dataset #####

input_data = pd.read_csv(f'{BASE_PATH}/data/protein_input_sets.csv')
data = list(input_data[['ProteinID', 'Sequence']].itertuples(index=False, name=None))
model_type = 'original'

######## Retrieve and save attn matrices #########

save_interval = 200
chunk_num = 0

batch_size = 2 # Or whatever fits your memory budget
all_results = {}
for i in range(0, len(data), batch_size):
    batch_data = data[i:i + batch_size]
    print(i)

    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, need_head_weights=True)

    attentions = results["attentions"]
    raw_sequence_lengths = torch.tensor([len(seq) for _, seq in batch_data])

    # Collect per-batch results into all_results
    for j, (label, sequence) in enumerate(zip(batch_labels, batch_strs)):
        actual_len = raw_sequence_lengths[j].item()
        all_results[label] = {
            'sequence': sequence,
            'attention_matrix': attentions[j].cpu().numpy(),
            'sequence_length': actual_len,
        }


    for obj in ['results', 'attentions', 'batch_tokens', 'attentions_tensor']:
        if obj in locals():
            del locals()[obj]

    torch.cuda.empty_cache()

     # Save and clear every save_interval proteins
    if len(all_results) >= save_interval:
        joblib.dump(all_results, f"{BASE_PATH}/attention_matrices_chunks/attention_matrices_labeled_{chunk_num}_{model_type}.joblib", compress=3)

        all_results = {}
        chunk_num += 1
        gc.collect()  # Force garbage collection

# Save final chunk
if all_results:
    # After all batches: save once
    joblib.dump(all_results, f"{BASE_PATH}/attention_matrices_chunks/attention_matrices_labeled_{chunk_num}_{model_type}.joblib", compress=3)

