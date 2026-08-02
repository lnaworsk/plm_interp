import mdtraj as md
import numpy as np
import pandas as pd
import os 
import torch 
import esm 
from io import BytesIO 
import tempfile
from tqdm import tqdm
import requests
import gc
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent 
BASE_PATH = SCRIPT_DIR.parent 

'''
Script to compute contact maps from MD trajectories and compare with contact maps derived from ESM-2 model Jacobians for disordered proteins.

code for do_apc, get_categorical_jacobian, and get_contacts is from https://github.com/zzhangzzhang/pLMs-interpretability/blob/main/jac/01_jac_calculate_visualise.ipynb

'''

def do_apc(x, rm=1):
  '''given matrix do apc correction'''
  # trying to remove different number of components
  # rm=0 remove none
  # rm=1 apc
  x = np.copy(x)
  if rm == 0:
    return x
  elif rm == 1:
    a1 = x.sum(0,keepdims=True) 
    a2 = x.sum(1,keepdims=True)
    y = x - (a1*a2)/x.sum()
  else:
    # decompose matrix, rm largest(s) eigenvectors
    u,s,v = np.linalg.svd(x)
    y = s[rm:] * u[:,rm:] @ v[rm:,:]
  np.fill_diagonal(y,0)
  return y

def get_categorical_jacobian(seq):
  # ∂in/∂out
  x,ln = alphabet.get_batch_converter()([("seq",seq)])[-1],len(seq)
  with torch.no_grad():
    f = lambda x: model(x)["logits"][...,1:(ln+1),4:24].cpu().numpy()
    fx = f(x.to(device))[0]
    x = torch.tile(x,[20,1]).to(device)
    fx_h = np.zeros((ln,20,ln,20))
    for n in range(ln): # for each position
      x_h = torch.clone(x)
      x_h[:,n+1] = torch.arange(4,24) # mutate to all 20 aa
      fx_h[n] = f(x_h)
    return fx_h - fx

# version of original function I modified which is optimized to reduce model forwarded passes and conserve memory
def get_categorical_jacobian_ln(seq, chunk_size = 150):
  x, ln = alphabet.get_batch_converter()([("seq", seq)])[-1], len(seq) # get tokenized input and seq length (not including BOS and EOS)
  with torch.no_grad():
    fx = model(x.to(device))["logits"][..., 1:(ln+1), 4:24].cpu().numpy()[0] # get the WT model logits for AA tokens only
    fx_h = np.zeros((ln, 20, ln, 20)) # create array to hold the mutated logits (jacobian)
    
    for start in range(0, ln, chunk_size): # process in chunks to save memory
        end = min(start + chunk_size, ln)
        chunk_len = end - start
        
        x_chunk = x.repeat(chunk_len * 20, 1).to(device) # create multiple copies of the original tokenized input
        mutations = torch.arange(4, 24, device=device).repeat(chunk_len) # lists all 20 mutations * the number of positions in the chunk to mutate 
        positions = torch.arange(start, end, device=device).repeat_interleave(20) + 1 # create list of token indices to mutate in the sequence
        x_chunk[torch.arange(chunk_len * 20, device=device), positions] = mutations # insert mutations at the correct positions
        
        fx_h[start:end] = model(x_chunk)["logits"][..., 1:(ln+1), 4:24].detach().cpu().numpy().reshape(chunk_len, 20, ln, 20)
    
    return fx_h - fx

def get_contacts(x, symm=True, center=True, rm=1):
  # convert jacobian (L,A,L,A) to contact map (L,L)
  j = x.copy()
  if center:
    for i in range(4): j -= j.mean(i,keepdims=True)
  j_fn = np.sqrt(np.square(j).sum((1,3))) #overall strength of functional coupling between residue i and residue j
  np.fill_diagonal(j_fn,0) # no self-interactions
  j_fn_corrected = do_apc(j_fn, rm=rm)
  if symm:
    j_fn_corrected = (j_fn_corrected + j_fn_corrected.T)/2
  return j_fn_corrected


# -------------------------------------
#           Load ESM-2 Model 
# -------------------------------------
md_method = 'frac' # either avg or frac
save_path = f'{BASE_PATH}/data/jacobian/jacobian_md_{md_method}.csv'
threshold = 0.65
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
model = model.eval()

# load IDRRome Sequences 
supp_table = pd.read_excel(f'{BASE_PATH}/data/Supplementary_Table_3.xlsx')
supp_table['seqlen'] = supp_table['sequence'].apply(len)
supp_table = supp_table[supp_table['seqlen'] <= 160]  # limit sequence length for compute time considerations
base_sid_url = "https://sid.erda.dk/share_redirect/AVZAJvJnCO/IDRome/" # base URL for MD trajectories and topologies

results = []
upper_dfs = []
save_every = 500  # save to disk every 500 iterations
directory = pd.read_csv(f'{BASE_PATH}/data/jac_batch_directory.csv')

for batch in directory.batch_num.unique():
  seq_names_batch = list(directory[directory['batch_num'] == batch]['seq_names'])
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


      # get contact map 
      distances, contacts = md.compute_contacts(traj, scheme = 'ca')
      contact_maps = md.geometry.squareform(distances, contacts)

      if md_method == 'avg': 
        avg_contact_map = np.mean(contact_maps, axis = 0)
        col_label = 'Distance'

      elif md_method == 'frac':
        n_res = contact_maps.shape[1] # Create a mask for which pairs were actually computed
        computed_mask = np.zeros((n_res, n_res), dtype=bool)
        
        # Mark pairs that were computed (from the contacts array)
        for pair in contacts:
            i, j = pair
            computed_mask[i, j] = True
            computed_mask[j, i] = True  # symmetric
        
        # Calculate binary contacts for all pairs
        binary_contacts = (contact_maps < threshold).astype(float)
        
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

      # -------------------------------------
      #           Compute Jacobian 
      # -------------------------------------

      # jacobian of the model

      jac = get_categorical_jacobian_ln(seq)
      for n in range(4): jac -= jac.mean(n,keepdims=True) 
      jac = (jac + jac.transpose(2,3,0,1))/2 
      jac_contacts = get_contacts(jac)

      jac_df = pd.DataFrame(jac_contacts)
      jac_df = jac_df.reset_index().melt(
          id_vars = "index", 
          var_name = "Residue_j", 
          value_name = "ContactStrength")

      jac_df = jac_df.rename(columns={"index": "Residue_i"})

  # -------------------------------------
  #               Correlations 
  # -------------------------------------

      df_merged = pd.merge(jac_df, md_df, on=["Residue_i", "Residue_j"])
      df_upper = df_merged[df_merged["Residue_i"] < df_merged["Residue_j"]]
      df_upper = df_upper.dropna() # remove the uncomputed contacts from md_traj
      pearson_corr = df_upper["ContactStrength"].corr(df_upper[col_label], method = "pearson")
      spearman_corr = df_upper["ContactStrength"].corr(df_upper[col_label], method = "spearman")

      # -------------------------------------
      #            Save Results 
      # ------------------------------------
    
      results.append({
          'seq_name': seq_name,
          'sequence': seq,
          'pearson_corr': pearson_corr,
          'spearman_corr': spearman_corr
      })

      upper_dfs.append(df_upper.assign(seq_name=seq_name))


      # periodically flush to disk
      if i % save_every == 0:
          # convert only the new chunk to a DataFrame
          chunk_df = pd.DataFrame(results)
          
          # append to CSV (no header if file exists)
          mode = 'a' if i > save_every else 'w'
          header = (i <= save_every)
          chunk_df.to_csv(save_path.replace('.csv', f'_batch{batch}.csv'), mode=mode, header=header, index=False)

          # ---- Save df_upper ----
          upper_path = save_path.replace('.csv', f'_batch{batch}_upper.csv')
          df_upper_chunk = pd.concat(upper_dfs, ignore_index=True)
          mode_upper = 'a' if os.path.exists(upper_path) else 'w'
          header_upper = not os.path.exists(upper_path)
          df_upper_chunk.to_csv(upper_path, mode=mode_upper, header=header_upper, index=False)

          # clear buffer to free memory
          results.clear()
          upper_dfs.clear()

          print(f"Saved {i} results so far...")

      del traj, distances, contacts, contact_maps, avg_contact_map, md_df
      del jac, jac_contacts, jac_df, df_merged, df_upper  
      gc.collect()

  # save any remaining rows at the end
  if results:
      pd.DataFrame(results).to_csv(save_path.replace('.csv', f'_batch{batch}.csv'), mode='a', header=not os.path.exists(save_path.replace('.csv', f'_batch{batch}.csv')), index=False)

  if upper_dfs:
      upper_path = save_path.replace('.csv', f'_batch{batch}_upper.csv')
      df_upper_chunk = pd.concat(upper_dfs, ignore_index=True)
      df_upper_chunk.to_csv(upper_path, mode='a', header=not os.path.exists(upper_path), index=False)


