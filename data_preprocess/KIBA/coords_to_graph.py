"""
Protein Coordinates to Graph Conversion

This script converts protein 3D coordinates to graph representations suitable for
graph neural network models. It processes pocket coordinates and generates geometric
features including RBF embeddings, positional encodings, and structural features.

Input:
    - KIBA_Protein_Domain_Alphfold3d_dict.pickle
    - KIBA_protein_domain_pocket_top*index_dict.pickle
    - KIBA_protein_domain_pocket_top*seq_dict.pickle

Output:
    - KIBA_Domain_DoGsite3_top*seqid_dict.pickle (intermediate)
    - KIBA_Domain_coord_graph_top*seqid_dict.pickle (final graph representation)
"""

import os
import numpy as np
import pickle
import torch
import math
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch_geometric
import torch.nn.functional as F
import torch_cluster
from tqdm import tqdm
import pandas as pd


def _normalize(tensor, dim=-1):
    """
    Normalizes a torch.Tensor along dimension `dim` without `nan`s.
    
    Args:
        tensor: Input tensor
        dim: Dimension along which to normalize
    
    Returns:
        Normalized tensor
    """
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """
    Radial Basis Function (RBF) embedding.
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    If `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    
    Args:
        D: Distance tensor
        D_min: Minimum distance for RBF centers
        D_max: Maximum distance for RBF centers
        D_count: Number of RBF centers
        device: Device to create tensors on
    
    Returns:
        RBF embedded tensor
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])  # D_mu=[1, D_count]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)  # D_expand=[edge_num, 1]

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)  # RBF=[edge_num, D_count]
    return RBF


# Amino acid letter to number mapping
letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                 'N': 2, 'Y': 18, 'M': 12, 'X': 20, '#': 21}

num_to_letter = {v: k for k, v in letter_to_num.items()}


def coord_to_graph(protein, max_seq_len, device):
    """
    Convert protein coordinates to geometric graph representation.
    
    Args:
        protein: Dictionary containing protein data (name, seq, coords)
        max_seq_len: Maximum sequence length to process
        device: Torch device (cuda or cpu)
    
    Returns:
        torch_geometric.data.Data: Graph representation of protein structure
    """
    top_k = 30
    num_rbf = 16
    max_seq_len = max_seq_len
    num_positional_embeddings = 16

    name = protein['name']
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], device=device, dtype=torch.float32)  
        coords = coords[:max_seq_len]
        # coords=[seq_len, 4, 3]  # Contains N, CA, C, O atom coordinates
        
        seq = torch.as_tensor([letter_to_num[a] for a in protein['seq']], 
                              device=device, dtype=torch.long)
        seq = seq[:max_seq_len]
        seq_len = torch.tensor([seq.shape[0]])
        # seq=[seq_len]
        
        # Create mask for finite coordinates
        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        # mask=[seq_len]
        
        coords[~mask] = np.inf  # Set infinite values for invalid coordinates
        # coords=[seq_len, 4, 3] 
        
        X_ca = coords[:, 1]  # Extract CA (alpha carbon) coordinates
        # X_ca=[seq_len, 3]
        
        # Build k-nearest neighbor graph
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)
        # edge_index=[2, (seq_len-infinite_num)*top_k]
        
        pos_embeddings = _positional_embeddings(edge_index, num_positional_embeddings, device)
        # pos_embeddings=[(seq_len-infinite_num)*top_k, num_positional_embeddings=16]
        
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]  # Edge vectors
        # E_vectors=[(seq_len-infinite_num)*top_k, 3]
        
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf, device=device)
        # Compute L2 norm of feature vectors and apply RBF
        # rbf=[(seq_len-infinite_num)*top_k, D_count=16]
        
        dihedrals = _dihedrals(coords)  # Dihedral angles (phi, psi, omega)
        # dihedrals=[seq_len, 6]
        
        orientations = _orientations(X_ca)  # CA orientation information between adjacent residues
        # orientations=[seq_len, 2, 3]
        
        sidechains = _sidechains(coords)  # Sidechain direction vectors
        # sidechains=[seq_len, 3]

        node_s = dihedrals  # Node scalar features
        # node_s=[seq_len, 6]
        
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)  # Node vector features
        # node_v=[seq_len, 3, 3]
        
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)  # Edge scalar features
        # edge_s=[(seq_len-infinite_num)*top_k, num_positional_embeddings+D_count=32]
        
        edge_v = _normalize(E_vectors).unsqueeze(-2)  # Edge vector features
        # edge_v=[(seq_len-infinite_num)*top_k, 1, 3]
        
        # Replace NaN values with 0
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, 
                                             (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(x=X_ca, seq=seq, seq_len=seq_len, name=name,
                                     node_s=node_s, node_v=node_v,
                                     edge_s=edge_s, edge_v=edge_v,
                                     edge_index=edge_index, mask=mask)
    return data


def _dihedrals(X, eps=1e-7):
    """
    Calculate dihedral angles of the protein backbone.
    These are the angles formed by adjacent triplets of residues (phi and psi angles).
    
    Args:
        X: Coordinate tensor [seq_len, 4, 3]
        eps: Small epsilon value for numerical stability
    
    Returns:
        Tensor containing dihedral angle features [seq_len, 6]
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    # X=[seq_len, 4, 3] 
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])  # X=[seq_len*3, 3]
    dX = X[1:] - X[:-1]  # dX=[seq_len*3-1, 3]
    U = _normalize(dX, dim=-1)  # U=[seq_len*3-1, 3]
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    # Contains representation of angles between adjacent residues in protein backbone
    return D_features


def _positional_embeddings(edge_index, num_embeddings=None, 
                           period_range=[2, 1000], num_positional_embeddings=16, 
                           device='cuda:0'):
    """
    Create positional embeddings for graph edges.
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_embeddings: Number of embedding dimensions
        period_range: Range for positional encoding periods
        num_positional_embeddings: Default number of positional embeddings
        device: Torch device
    
    Returns:
        Positional embedding tensor
    """
    num_embeddings = num_embeddings or num_positional_embeddings
    d = edge_index[0] - edge_index[1]  # Relative positions between nodes
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )  # Frequency tensor for positional encoding
    
    angles = d.unsqueeze(-1) * frequency  # Calculate angles from relative positions
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)  # Concatenate cos and sin
    return E


def _orientations(X):
    """
    Calculate orientations between adjacent residues in the protein backbone.
    Captures both forward and backward directions.
    
    Args:
        X: CA atom coordinates [seq_len, 3]
    
    Returns:
        Orientation tensor [seq_len, 2, 3]
    """
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    """
    Calculate sidechain direction vectors.
    The sidechain is the branch structure formed by atoms on the backbone 
    (typically CA, C, N atoms).
    
    Args:
        X: Coordinate tensor [seq_len, 4, 3]
    
    Returns:
        Sidechain direction vectors [seq_len, 3]
    """
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def process_top_n(top_n, process_csv='process.csv', 
                  protein_3d_dict_file='KIBA_Protein_Domain_Alphfold3d_dict.pickle',
                  device='cuda:0'):
    """
    Process protein pockets and convert to graph representations for top-N predictions.
    
    Args:
        top_n: Which top-N prediction to process (1, 2, or 3)
        process_csv: CSV file with protein IDs
        protein_3d_dict_file: Pickle file with 3D protein structures
        device: Device for torch computations
    
    Returns:
        None (saves pickle files)
    """
    # Read protein IDs
    df = pd.read_csv(process_csv)
    target_key = list(df['target_key'].unique())

    # Load dictionaries
    with open(protein_3d_dict_file, 'rb') as f:
        alpha3d_dict = pickle.load(f)
    
    with open(f'KIBA_protein_domain_pocket_top{top_n}index_dict.pickle', 'rb') as f:
        pocket_dict = pickle.load(f)
    
    with open(f'KIBA_protein_domain_pocket_top{top_n}seq_dict.pickle', 'rb') as f:
        seqnew_dict = pickle.load(f)

    # Extract pocket coordinates
    pocket_coords_dict = {}
    for key in tqdm(target_key, desc=f"Processing top{top_n} coordinates"):
        if key not in pocket_dict:
            continue
            
        resn = pocket_dict[key]
        coords = {}
        begin_resn = alpha3d_dict[key]['begin_resn']
        resn = [i - begin_resn for i in resn]
        
        coords['coords'] = alpha3d_dict[key]['coords'][resn]
        coords['seq'] = seqnew_dict[key]
        coords['name'] = alpha3d_dict[key]['name']
        pocket_coords_dict[key] = coords

    # Save intermediate pocket coordinates
    with open(f'KIBA_Domain_DoGsite3_top{top_n}seqid_dict.pickle', 'wb') as f:
        pickle.dump(pocket_coords_dict, f)

    # Convert coordinates to graphs
    coords_graph = {}
    for key in tqdm(target_key, desc=f"Converting top{top_n} to graphs"):
        if key not in pocket_coords_dict:
            continue
            
        coord = pocket_coords_dict[key]
        coord_graph = coord_to_graph(coord, max_seq_len=120, device=device)
        coords_graph[key] = coord_graph

    # Print sample output
    if len(coords_graph) > 0:
        sample_key = list(coords_graph.keys())[0]
        print(f"\nSample graph for {sample_key}:")
        print(coords_graph[sample_key])

    # Save final graph representations
    output_file = f'KIBA_Domain_coord_graph_top{top_n}seqid_dict.pickle'
    with open(output_file, 'wb') as f:
        pickle.dump(coords_graph, f)
    
    print(f"\nSuccessfully processed {len(coords_graph)} proteins for top{top_n}.")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    # Process all three top-N predictions
    # Make sure you have CUDA available or change device to 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    for top_n in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Processing Top {top_n} Predictions")
        print(f"{'='*60}")
        process_top_n(top_n, device=device)
