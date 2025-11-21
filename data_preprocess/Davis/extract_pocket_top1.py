"""
Pocket Extraction - Top 1 Prediction for Davis Dataset

This script extracts binding pocket residue indices and sequences from 
the top 1 predicted binding site PDB files.

Input:
    - Davis_Protein_Domain_Alphfold3d_dict.pickle
    - PDB files in Domain_Key_Pocket_files_top1/

Output:
    - Davis_protein_domain_pocket_top1index_dict.pickle
    - Davis_protein_domain_pocket_top1len_dict.pickle
    - Davis_protein_domain_pocket_top1seq_dict.pickle
"""

import os
import pickle
import pandas as pd
from tqdm import tqdm

# Configuration - Update these paths according to your setup
PROCESS_CSV = 'process.csv'
PROTEIN_3D_DICT = 'Davis_Protein_Domain_Alphfold3d_dict.pickle'
POCKET_PDB_DIR = 'Domain_Key_Pocket_files_top1'

# Read protein IDs
df = pd.read_csv(PROCESS_CSV)
target_key = list(df['target_key'].unique())

# Load protein 3D structure dictionary
with open(PROTEIN_3D_DICT, 'rb') as file:
    protein_3D_dict = pickle.load(file)

def pdb_res_index(pdb_file):
    """
    Extract residue indices from PDB file.
    
    Args:
        pdb_file: Path to PDB file
    
    Returns:
        list: Sorted list of unique residue numbers
    """
    resns = []
    for line in open(pdb_file, 'rb'):
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
    pdb_file = os.path.join(POCKET_PDB_DIR, f'{key}.pdb')
    
    if not os.path.exists(pdb_file):
        print(f"Warning: PDB file not found for {key}")
        continue
    
    resns = pdb_res_index(pdb_file)  # Residue numbering starts from 1
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
with open('Davis_protein_domain_pocket_top1index_dict.pickle', 'wb') as f:
    pickle.dump(proteins_pocket_index_dict, f)

with open('Davis_protein_domain_pocket_top1len_dict.pickle', 'wb') as f:
    pickle.dump(proteins_pocket_indexlen_dict, f)

with open('Davis_protein_domain_pocket_top1seq_dict.pickle', 'wb') as f:
    pickle.dump(proteins_pocket_seq_dict, f)

print(f'\nStatistics:')
print(f'Max sequence length: {max(resns_len)}')
print(f'Min sequence length: {min(resns_len)}')
print(f'Total proteins processed: {len(proteins_pocket_index_dict)}')
print('\n# Davis dataset: max 49, min 9 (domain)')
