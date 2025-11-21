"""
Pocket Extraction - Top 3 Predictions for Davis Dataset

This script extracts binding pocket residue indices and sequences from 
the combined top 1, top 2, and top 3 predicted binding site PDB files.

Input:
    - Davis_Protein_Domain_Alphfold3d_dict.pickle
    - PDB files in Domain_Key_Pocket_files_top1/, top2/, and top3/

Output:
    - Davis_protein_domain_pocket_top3index_dict.pickle
    - Davis_protein_domain_pocket_top3len_dict.pickle
    - Davis_protein_domain_pocket_top3seq_dict.pickle
"""

import os
import pickle
import pandas as pd
from tqdm import tqdm

# Configuration - Update these paths according to your setup
PROCESS_CSV = 'process.csv'
PROTEIN_3D_DICT = 'Davis_Protein_Domain_Alphfold3d_dict.pickle'
POCKET_PDB_DIR_TOP1 = 'Domain_Key_Pocket_files_top1'
POCKET_PDB_DIR_TOP2 = 'Domain_Key_Pocket_files_top2'
POCKET_PDB_DIR_TOP3 = 'Domain_Key_Pocket_files_top3'

# Read protein IDs
df = pd.read_csv(PROCESS_CSV)
target_key = list(df['target_key'].unique())

# Load protein 3D structure dictionary
with open(PROTEIN_3D_DICT, 'rb') as file:
    protein_3D_dict = pickle.load(file)

def pdb_res_index(pdb_file_top1, pdb_file_top2, pdb_file_top3):
    """
    Extract residue indices from three PDB files (top1, top2, and top3 predictions).
    
    Args:
        pdb_file_top1: Path to top1 PDB file
        pdb_file_top2: Path to top2 PDB file
        pdb_file_top3: Path to top3 PDB file
    
    Returns:
        list: Sorted list of unique residue numbers from all three files
    """
    resns = []
    
    # Read top1 file
    for line in open(pdb_file_top1, 'rb'):
        line = line.decode("utf-8", "ignore").rstrip()
        if line[:4] == 'ATOM':
            resn = line[22:22+5].strip()
            resns.append(int(resn))
    
    # Read top2 file
    for line in open(pdb_file_top2, 'rb'):
        line = line.decode("utf-8", "ignore").rstrip()
        if line[:4] == 'ATOM':
            resn = line[22:22+5].strip()
            resns.append(int(resn))
    
    resns = sorted(list(set(resns)))
    
    # Read top3 file
    for line in open(pdb_file_top3, 'rb'):
        line = line.decode("utf-8", "ignore").rstrip()
        if line[:4] == 'ATOM':
            resn = line[22:22+5].strip()
            resns.append(int(resn))
    
    resns = sorted(list(set(resns)))   
    return resns

# Initialize dictionaries to store results
proteins_pocket_index_dict = {}
proteins_pocket_indexlen_dict = {}   
proteins_pocket_seq_dict = {}
resns_len = []

# Process each protein
for key in tqdm(target_key, desc="Extracting pockets"):
    pdb_file_top1 = os.path.join(POCKET_PDB_DIR_TOP1, f'{key}.pdb')
    pdb_file_top2 = os.path.join(POCKET_PDB_DIR_TOP2, f'{key}.pdb')
    pdb_file_top3 = os.path.join(POCKET_PDB_DIR_TOP3, f'{key}.pdb')
    
    if not all([os.path.exists(f) for f in [pdb_file_top1, pdb_file_top2, pdb_file_top3]]):
        print(f"Warning: PDB file not found for {key}")
        continue
    
    resns = pdb_res_index(pdb_file_top1, pdb_file_top2, pdb_file_top3)
    begin_resn = protein_3D_dict[key]['begin_resn']
    
    print(f"{key}, begin_resn: {begin_resn}")
    
    # Extract pocket sequence
    proteins_pocket_seq_dict[key] = ''.join([
        protein_3D_dict[key]['seq'][i-begin_resn] for i in resns
    ])
    proteins_pocket_index_dict[key] = resns
    proteins_pocket_indexlen_dict[key] = len(resns)
    resns_len.append(len(resns))

# Save results to pickle files
with open('Davis_protein_domain_pocket_top3index_dict.pickle', 'wb') as f:
    pickle.dump(proteins_pocket_index_dict, f)

with open('Davis_protein_domain_pocket_top3len_dict.pickle', 'wb') as f:
    pickle.dump(proteins_pocket_indexlen_dict, f)

with open('Davis_protein_domain_pocket_top3seq_dict.pickle', 'wb') as f:
    pickle.dump(proteins_pocket_seq_dict, f)

print(f'\nStatistics:')
print(f'Max sequence length: {max(resns_len)}')
print(f'Min sequence length: {min(resns_len)}')
print(f'Total proteins processed: {len(proteins_pocket_index_dict)}')
print('\n# Davis dataset: max 76, min 23 (domain)')
