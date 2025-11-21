"""
AlphaFold PDB Data Preprocessing Script for Davis Dataset

This script processes PDB files from AlphaFold and extracts protein structural information
including coordinates, sequences, and residue numbering.

Output: Davis_Protein_Domain_Alphfold3d_dict.pickle
"""

import numpy as np
import os
from tqdm import tqdm
import json
import pickle


# Amino acid mappings
alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def N_to_AA(x):
    """
    Convert numerical amino acid encoding to letter representation.
    Args:
        x: Array of numerical amino acid codes
    Returns:
        List of amino acid sequences as strings
    """
    x = np.array(x)
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

atoms = ['N','CA','C']

def pdb_coords_seq(x, atoms=['N','CA','C'], chain=None):
    """
    Extract coordinates and sequence from PDB file.
    
    Args:
        x: Path to PDB file
        atoms: List of atom types to extract (default: ['N','CA','C'])
        chain: Specific chain to extract (None for all chains)
    
    Returns:
        tuple: (coordinates array, sequence, minimum residue number)
    """
    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()
        
        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12+4].strip()
                resi = line[17:17+3]
                resn = line[22:22+5].strip()
                x, y, z = [float(line[i:(i+8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1])-1
                else:
                    resa, resn = "", int(resn)-1
                    
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                    
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])
    
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn+1):
            if resn in seq:
                for k in sorted(seq[resn]): 
                    seq_.append(aa_3_N.get(seq[resn][k], 20))  # Encode protein sequence
            else: 
                seq_.append(20)
                
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]: 
                            xyz_.append(xyz[resn][k][atom])
                        else: 
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms: 
                    xyz_.append(np.full(3, np.nan))
                
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)), min_resn
    except TypeError:
        return 'no_chain', 'no_chain'

def pdb_coords_dict(pdb_path, pdb_id):
    """
    Create a dictionary containing protein structural information.
    
    Args:
        pdb_path: Directory containing PDB files
        pdb_id: Protein ID
    
    Returns:
        dict: Dictionary with protein sequence, coordinates, and residue numbering
    """
    pdb_file = os.path.join(pdb_path, f"{pdb_id}.pdb")
    protein_dict = {}
    protein_dict['seq'] = []
    xyz, seq, min_resn = pdb_coords_seq(pdb_file, atoms=['N','CA','C','O'])
    protein_dict['seq'] = seq[0]
    protein_dict['name'] = pdb_id
    protein_dict['coords'] = xyz
    protein_dict['begin_resn'] = min_resn + 1 
    return protein_dict


if __name__ == "__main__":
    import pandas as pd
    
    # Configuration - Update these paths according to your setup
    PROCESS_CSV = 'process.csv'  # CSV file containing target protein IDs
    PDB_PATH = 'prot_3d_for_Davis'  # Directory containing PDB files
    OUTPUT_FILE = 'Davis_Protein_Domain_Alphfold3d_dict.pickle'
    
    # Read protein IDs from CSV
    df = pd.read_csv(PROCESS_CSV)
    target_key = list(df['target_key'].unique())
    
    # Process all proteins
    proteins_dict = {}
    for pdb_id in tqdm(target_key, desc="Processing proteins"):
        protein_dict = pdb_coords_dict(pdb_path=PDB_PATH, pdb_id=pdb_id)
        proteins_dict[pdb_id] = protein_dict
    
    # Save results
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(proteins_dict, f)
    
    print(f"\nSuccessfully processed {len(proteins_dict)} proteins.")
    print(f"Output saved to: {OUTPUT_FILE}")
