import mdtraj as md
import numpy as np
import pandas as pd
import os 
import torch 
from io import BytesIO 
import tempfile
from tqdm import tqdm
import requests
import gc
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 
'''
Script to compute md-derived contact maps using variable interaction threshold according to: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005941#sec002
sigma_ij values are found below and additionally utilized from the same source 
'''

sigma_map = {
    'A': 5.04,  # ALA
    'R': 6.56,  # ARG
    'N': 5.68,  # ASN
    'D': 5.58,  # ASP
    'C': 5.48,  # CYS
    'Q': 6.02,  # GLN
    'E': 5.92,  # GLU
    'G': 4.50,  # GLY
    'H': 6.08,  # HIS
    'I': 6.18,  # ILE
    'L': 6.18,  # LEU
    'K': 6.36,  # LYS
    'M': 6.18,  # MET
    'F': 6.36,  # PHE
    'P': 5.56,  # PRO
    'S': 5.18,  # SER
    'T': 5.62,  # THR
    'W': 6.78,  # TRP
    'Y': 6.46,  # TYR
    'V': 5.86,  # VAL
}
save_path = f'{BASE_PATH}/data/jacobian/md_variable_sigma.csv'

# load IDRRome Sequences 
supp_table = pd.read_excel(f'{BASE_PATH}/data/Supplementary_Table_3.xlsx')
supp_table['seqlen'] = supp_table['sequence'].apply(len)
supp_table = supp_table[supp_table['seqlen'] <= 160]  # limit sequence length for compute time considerations
base_sid_url = "https://sid.erda.dk/share_redirect/AVZAJvJnCO/IDRome/" # base URL for MD trajectories and topologies

upper_dfs = []
save_every = 500  # save to disk every 500 iterations
directory = pd.read_csv(f'{BASE_PATH}/data/jac_batch_directory.csv')

for batch in directory.batch_num.unique():
    seq_names_batch = list(directory[directory['batch_num'] == batch]['seq_names'])
    save_path_batch = save_path.replace('.csv', f'_batch{batch}.csv')
    for i, seq_name in enumerate(tqdm(seq_names_batch), start=1):
        seq = supp_table[supp_table['seq_name'] == seq_name]['sequence'].values[0]

      # -------------------------------------
      #     Get Contact Maps from MD Traj 
      # -------------------------------------

      # Split seq_name into UniProt ID and residue range
        uniprot_id, res_start, res_end = seq_name.split('_')    
        p1, p2, p3, rest = uniprot_id[:2], uniprot_id[2:4], uniprot_id[4:6], uniprot_id[6:]
        if len(rest) > 0:
            p3 += '/' + rest
            
        folder_path = f"{p1}/{p2}/{p3}/{res_start}_{res_end}/"
        traj_url = base_sid_url + folder_path + "traj.xtc"
        top_url = base_sid_url + folder_path + "top.pdb"

        # Download files
        traj_req = requests.get(traj_url)
        top_req  = requests.get(top_url)

      # Save temporarily
        with tempfile.NamedTemporaryFile(suffix=".xtc", delete=False) as tmp_traj, \
            tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_top:

            tmp_traj.write(traj_req.content)
            tmp_top.write(top_req.content)
            tmp_traj_filename = tmp_traj.name
            tmp_top_filename  = tmp_top.name

        # Load MDTraj trajectory
        try:
            traj = md.load(tmp_traj_filename, top=tmp_top_filename)
        except Exception as e:
            continue

        # Delete temporary files
        os.remove(tmp_traj_filename)
        os.remove(tmp_top_filename)
        
        sigmas = np.array([sigma_map.get(aa) for aa in seq])
        sigma_ij = (sigmas[:, None] + sigmas[None, :]) / 2.0
        cutoff_matrix = (2 ** (1/6) * sigma_ij) / 10.0
        
        distances, contacts = md.compute_contacts(traj, scheme = 'ca')
        contact_maps = md.geometry.squareform(distances, contacts)
        n_res = contact_maps.shape[1] # Create a mask for which pairs were actually computed
        computed_mask = np.zeros((n_res, n_res), dtype=bool)
        # Mark pairs that were computed (from the contacts array)
        for pair in contacts:  
            i, j = pair
            computed_mask[i, j] = True
            computed_mask[j, i] = True  # symmetric
            
        # Calculate binary contacts for all pairs
        binary_contacts = (contact_maps < cutoff_matrix[np.newaxis, :, :]).astype(float)
            
        # Set uncomputed pairs to NaN
        binary_contacts[:, ~computed_mask] = np.nan
            
        # Average across frames, ignoring NaN values (but preserving NaN for uncomputed pairs)
        avg_contact_map = np.nanmean(binary_contacts, axis=0)
        col_label = 'Frac'

        n_res = avg_contact_map.shape[0]

        data = []
        for k in range(n_res):
            for j in range(n_res):
                data.append({
                    'Residue_i': k,
                    'Residue_j': j,
                    col_label:  avg_contact_map[k, j]
                })

        md_df = pd.DataFrame(data)
        del data

        df_upper = md_df[md_df["Residue_i"] < md_df["Residue_j"]]
        df_upper = df_upper.dropna() # remove the uncomputed contacts from md_traj
        upper_dfs.append(df_upper.assign(seq_name=seq_name))


        # periodically flush to disk
        if i % save_every == 0:
            
            # ---- Save df_upper ----
            df_upper_chunk = pd.concat(upper_dfs, ignore_index=True)
            mode_upper = 'a' if os.path.exists(save_path_batch) else 'w'
            header_upper = not os.path.exists(save_path_batch)
            df_upper_chunk.to_csv(save_path_batch, mode=mode_upper, header=header_upper, index=False)

            # clear buffer to free memory
            upper_dfs.clear()

            print(f"Saved {i} results so far...")

        del traj, distances, contacts, contact_maps, avg_contact_map, md_df
        gc.collect()

    if upper_dfs:
        df_upper_chunk = pd.concat(upper_dfs, ignore_index=True)
        df_upper_chunk.to_csv(save_path_batch, mode='a', header=not os.path.exists(save_path_batch), index=False)


