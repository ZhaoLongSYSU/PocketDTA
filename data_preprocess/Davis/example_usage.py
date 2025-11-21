"""
Example Usage of Davis Pocket Graph Data

This script demonstrates how to load and use the generated graph pickle files
for machine learning applications.
"""

import pickle
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def load_graph_data(pickle_file):
    """
    Load graph data from pickle file.
    
    Args:
        pickle_file: Path to pickle file
    
    Returns:
        Dictionary of protein graphs
    """
    print(f"Loading {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        graph_dict = pickle.load(f)
    print(f"Loaded {len(graph_dict)} protein graphs")
    return graph_dict


def inspect_graph(graph_dict, protein_id=None):
    """
    Inspect a single graph structure.
    
    Args:
        graph_dict: Dictionary of protein graphs
        protein_id: Specific protein ID to inspect (or None for first)
    """
    if protein_id is None:
        protein_id = list(graph_dict.keys())[0]
    
    graph = graph_dict[protein_id]
    
    print(f"\n{'='*60}")
    print(f"Protein ID: {protein_id}")
    print(f"{'='*60}")
    print(f"\nGraph Structure:")
    print(f"  - Number of nodes: {graph.x.shape[0]}")
    print(f"  - Number of edges: {graph.edge_index.shape[1]}")
    print(f"  - Sequence length: {graph.seq_len.item()}")
    print(f"\nNode Features:")
    print(f"  - Scalar features shape: {graph.node_s.shape}")
    print(f"  - Vector features shape: {graph.node_v.shape}")
    print(f"\nEdge Features:")
    print(f"  - Scalar features shape: {graph.edge_s.shape}")
    print(f"  - Vector features shape: {graph.edge_v.shape}")
    print(f"\nSequence Preview:")
    seq_str = ''.join([num_to_letter.get(i.item(), 'X') for i in graph.seq[:20]])
    print(f"  {seq_str}...")
    
    return graph


# Amino acid number to letter mapping
num_to_letter = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G',
    8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P',
    15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V', 20: 'X', 21: '#'
}


def create_dataloader(graph_dict, batch_size=32, shuffle=True):
    """
    Create PyTorch Geometric DataLoader.
    
    Args:
        graph_dict: Dictionary of protein graphs
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader
    """
    graphs = list(graph_dict.values())
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)
    print(f"\nCreated DataLoader with {len(graphs)} graphs")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(loader)}")
    return loader


def statistics(graph_dict):
    """
    Compute statistics across all graphs.
    
    Args:
        graph_dict: Dictionary of protein graphs
    """
    num_nodes = []
    num_edges = []
    seq_lengths = []
    
    for protein_id, graph in graph_dict.items():
        num_nodes.append(graph.x.shape[0])
        num_edges.append(graph.edge_index.shape[1])
        seq_lengths.append(graph.seq_len.item())
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"\nNumber of proteins: {len(graph_dict)}")
    print(f"\nNode counts:")
    print(f"  - Mean: {np.mean(num_nodes):.2f}")
    print(f"  - Std:  {np.std(num_nodes):.2f}")
    print(f"  - Min:  {np.min(num_nodes)}")
    print(f"  - Max:  {np.max(num_nodes)}")
    print(f"\nEdge counts:")
    print(f"  - Mean: {np.mean(num_edges):.2f}")
    print(f"  - Std:  {np.std(num_edges):.2f}")
    print(f"  - Min:  {np.min(num_edges)}")
    print(f"  - Max:  {np.max(num_edges)}")
    print(f"\nSequence lengths:")
    print(f"  - Mean: {np.mean(seq_lengths):.2f}")
    print(f"  - Std:  {np.std(seq_lengths):.2f}")
    print(f"  - Min:  {np.min(seq_lengths)}")
    print(f"  - Max:  {np.max(seq_lengths)}")
    
    return num_nodes, num_edges, seq_lengths


def plot_distributions(num_nodes, num_edges, seq_lengths, save_path=None):
    """
    Plot distributions of graph properties.
    
    Args:
        num_nodes: List of node counts
        num_edges: List of edge counts
        seq_lengths: List of sequence lengths
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Node count distribution
    axes[0].hist(num_nodes, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Nodes')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Node Count Distribution')
    axes[0].axvline(np.mean(num_nodes), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(num_nodes):.1f}')
    axes[0].legend()
    
    # Edge count distribution
    axes[1].hist(num_edges, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Number of Edges')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Edge Count Distribution')
    axes[1].axvline(np.mean(num_edges), color='red', linestyle='--',
                    label=f'Mean: {np.mean(num_edges):.1f}')
    axes[1].legend()
    
    # Sequence length distribution
    axes[2].hist(seq_lengths, bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Sequence Length Distribution')
    axes[2].axvline(np.mean(seq_lengths), color='red', linestyle='--',
                    label=f'Mean: {np.mean(seq_lengths):.1f}')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def example_batch_processing(loader):
    """
    Demonstrate batch processing with DataLoader.
    
    Args:
        loader: PyTorch Geometric DataLoader
    """
    print(f"\n{'='*60}")
    print(f"Example Batch Processing")
    print(f"{'='*60}")
    
    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  - Batch size: {batch.num_graphs}")
        print(f"  - Total nodes in batch: {batch.x.shape[0]}")
        print(f"  - Total edges in batch: {batch.edge_index.shape[1]}")
        print(f"  - Node scalar features: {batch.node_s.shape}")
        print(f"  - Edge scalar features: {batch.edge_s.shape}")
        
        if batch_idx >= 2:  # Show only first 3 batches
            print(f"\n... ({len(loader) - 3} more batches)")
            break


def compare_top_n(top1_file, top2_file, top3_file):
    """
    Compare statistics across top-1, top-2, and top-3 predictions.
    
    Args:
        top1_file: Path to top1 pickle file
        top2_file: Path to top2 pickle file
        top3_file: Path to top3 pickle file
    """
    print(f"\n{'='*60}")
    print(f"Comparing Top-N Predictions")
    print(f"{'='*60}")
    
    for n, file in enumerate([top1_file, top2_file, top3_file], 1):
        try:
            graph_dict = load_graph_data(file)
            num_nodes, _, _ = statistics(graph_dict)
            print(f"\nTop-{n} average pocket size: {np.mean(num_nodes):.2f} residues")
        except FileNotFoundError:
            print(f"\nTop-{n} file not found: {file}")


def main():
    """Main demonstration function."""
    print("="*60)
    print("Davis Pocket Graph Data - Example Usage")
    print("="*60)
    
    # Example 1: Load and inspect a single graph
    print("\n[Example 1] Loading and inspecting graph data...")
    try:
        graph_dict = load_graph_data('Davis_Domain_coord_graph_top1seqid_dict.pickle')
        sample_graph = inspect_graph(graph_dict)
    except FileNotFoundError:
        print("Error: Pickle file not found. Please run the pipeline first.")
        return
    
    # Example 2: Compute statistics
    print("\n[Example 2] Computing dataset statistics...")
    num_nodes, num_edges, seq_lengths = statistics(graph_dict)
    
    # Example 3: Plot distributions
    print("\n[Example 3] Plotting distributions...")
    try:
        plot_distributions(num_nodes, num_edges, seq_lengths, 
                          save_path='graph_statistics.png')
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Example 4: Create DataLoader
    print("\n[Example 4] Creating DataLoader...")
    loader = create_dataloader(graph_dict, batch_size=32, shuffle=True)
    
    # Example 5: Process batches
    print("\n[Example 5] Processing batches...")
    example_batch_processing(loader)
    
    # Example 6: Compare top-N predictions
    print("\n[Example 6] Comparing top-N predictions...")
    compare_top_n(
        'Davis_Domain_coord_graph_top1seqid_dict.pickle',
        'Davis_Domain_coord_graph_top2seqid_dict.pickle',
        'Davis_Domain_coord_graph_top3seqid_dict.pickle'
    )
    
    print(f"\n{'='*60}")
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
