# KIBA Protein-Drug Binding Pocket Analysis

This repository contains scripts for processing protein structures from the KIBA dataset, extracting binding pockets using DoGSite3, and converting them to graph representations for machine learning applications.

## Overview

The pipeline consists of the following steps:

1. **Download AlphaFold structures** - Retrieve PDB files from AlphaFold database
2. **Preprocess protein data** - Extract coordinates and sequences from PDB files
3. **Predict binding pockets** - Use DoGSite3 to predict binding sites
4. **Extract pocket residues** - Parse pocket PDB files for top 1/2/3 predictions
5. **Convert to graphs** - Transform 3D coordinates into geometric graph representations

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `torch` - PyTorch for graph processing
- `torch-geometric` - Geometric deep learning
- `torch-cluster` - Graph clustering operations
- `selenium` - Web scraping for DoGSite3
- `tqdm` - Progress bars
- `pickle` - Data serialization

## Project Structure

```
data_preprocess/KIBA/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore patterns
├── PROJECT_SUMMARY.md                 # Technical overview and specifications
│
├── config_template.py                 # Configuration template
├── example_process.csv                # Example protein ID file
│
├── download_alphafold_pdb.py         # Download PDB files from AlphaFold
├── preprocess_alphafold_data.py      # Extract protein structural data
├── scrape_dogsite_pockets.py         # Scrape DoGSite3 for pocket predictions
├── extract_pocket_top1.py            # Extract top 1 pocket residues
├── extract_pocket_top2.py            # Extract top 1+2 pocket residues
├── extract_pocket_top3.py            # Extract top 1+2+3 pocket residues
├── coords_to_graph.py                # Convert coordinates to graphs
│
├── run_pipeline.py                   # Complete workflow script
└── example_usage.py                  # Example code for using generated data
```

## Input Data

You need to prepare the following input files:

1. **process.csv** - CSV file containing a column `target_key` with protein UniProt IDs
2. **KIBA_protein_id.txt** - Text file with protein IDs (one per line)

Example `process.csv`:
```csv
target_key
P00519
Q13464
P04626
...
```

## Usage

### Option 1: Run Complete Pipeline

```bash
python run_pipeline.py
```

This will execute all steps sequentially. Make sure to update the configuration paths in the script.

### Option 2: Run Individual Steps

#### Step 1: Download AlphaFold PDB Files

```bash
python download_alphafold_pdb.py
```

Update the following variables in the script:
- `PROTEIN_ID_FILE`: Path to your protein ID list
- `TARGET_FOLDER`: Where to save PDB files
- `DOWNLOAD_FOLDER`: Your browser's download directory

#### Step 2: Preprocess Protein Data

```bash
python preprocess_alphafold_data.py
```

Update configuration:
- `PROCESS_CSV`: Path to process.csv
- `PDB_PATH`: Directory containing downloaded PDB files

**Output**: `KIBA_Protein_Domain_Alphfold3d_dict.pickle`

#### Step 3: Predict Binding Pockets (DoGSite3)

```bash
python scrape_dogsite_pockets.py
```

This script uses Selenium to automate pocket prediction on https://proteins.plus/

**Note**: This step can take several hours. You'll need to manually extract the pocket PDB files from the downloaded ZIP files.

#### Step 4: Extract Pocket Residues

```bash
python extract_pocket_top1.py
python extract_pocket_top2.py
python extract_pocket_top3.py
```

**Outputs**:
- `KIBA_protein_domain_pocket_top{1,2,3}index_dict.pickle`
- `KIBA_protein_domain_pocket_top{1,2,3}len_dict.pickle`
- `KIBA_protein_domain_pocket_top{1,2,3}seq_dict.pickle`

#### Step 5: Convert to Graph Representations

```bash
python coords_to_graph.py
```

**Final Outputs**:
- `KIBA_Domain_coord_graph_top1seqid_dict.pickle`
- `KIBA_Domain_coord_graph_top2seqid_dict.pickle`
- `KIBA_Domain_coord_graph_top3seqid_dict.pickle`

These files contain geometric graph representations with:
- Node features: Dihedral angles, orientations, sidechain vectors
- Edge features: RBF embeddings, positional encodings
- Graph structure: k-NN graph based on CA atom distances

## Graph Representation Details

The graph conversion includes:

### Node Features
- **Scalar features** (6D): Dihedral angles (phi, psi, omega) as cos/sin pairs
- **Vector features** (3×3D): Backbone orientations + sidechain directions

### Edge Features
- **Scalar features** (32D): RBF distance embeddings (16D) + positional encodings (16D)
- **Vector features** (1×3D): Normalized edge direction vectors

### Graph Structure
- **Nodes**: CA (alpha carbon) atoms of pocket residues
- **Edges**: k-NN graph with k=30 neighbors

## Output Format

Each pickle file contains a dictionary:
```python
{
    'PROTEIN_ID': torch_geometric.data.Data(
        x=...,          # CA coordinates [N, 3]
        seq=...,        # Amino acid sequence [N]
        seq_len=...,    # Sequence length
        name=...,       # Protein name
        node_s=...,     # Node scalar features [N, 6]
        node_v=...,     # Node vector features [N, 3, 3]
        edge_s=...,     # Edge scalar features [E, 32]
        edge_v=...,     # Edge vector features [E, 1, 3]
        edge_index=..., # Edge connectivity [2, E]
        mask=...        # Valid residue mask [N]
    ),
    ...
}
```

## Configuration

All scripts have a configuration section at the top. Update these paths before running:

```python
# Example configuration
PROCESS_CSV = 'process.csv'
PDB_PATH = 'prot_3d_for_KIBA'
OUTPUT_FILE = 'output.pickle'
```

You can also use the `config_template.py` file as a reference. Copy it to `config.py` and update the paths:

```bash
cp config_template.py config.py
# Then edit config.py with your paths
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**Important**: PyTorch Geometric requires special installation. See [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

### 2. Prepare Input Files

Create `process.csv` with protein IDs:
```csv
target_key
P00519
Q13464
P04626
```

### 3. Run the Pipeline

```bash
# Check prerequisites
python run_pipeline.py --step check

# Run complete pipeline
python run_pipeline.py --step all
```

## Example Usage

After generating the pickle files, you can use them in your ML models:

```python
import pickle
import torch
from torch_geometric.loader import DataLoader

# Load graph data
with open('KIBA_Domain_coord_graph_top1seqid_dict.pickle', 'rb') as f:
    graph_dict = pickle.load(f)

print(f"Number of proteins: {len(graph_dict)}")

# Inspect a sample graph
protein_id = list(graph_dict.keys())[0]
graph = graph_dict[protein_id]
print(f"\nProtein: {protein_id}")
print(f"Nodes: {graph.x.shape[0]}")
print(f"Edges: {graph.edge_index.shape[1]}")

# Create DataLoader for training
graphs = list(graph_dict.values())
loader = DataLoader(graphs, batch_size=32, shuffle=True)

for batch in loader:
    # Your model training code here
    node_features = batch.node_s  # [N, 6]
    edge_features = batch.edge_s  # [E, 32]
    break
```

See `example_usage.py` for more detailed examples including visualization.

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Edit `coords_to_graph.py` and change device to CPU:
```python
device = 'cpu'  # Instead of 'cuda:0'
```

### Issue: "Selenium can't find element"
**Solution**: Increase wait times in `scrape_dogsite_pockets.py`:
```python
time.sleep(10)  # Instead of time.sleep(5)
```

### Issue: "ModuleNotFoundError: No module named 'torch_geometric'"
**Solution**: Install PyTorch Geometric following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html):

For CUDA 11.3:
```bash
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

For CPU only:
```bash
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
```

### Issue: "File not found" errors
**Solution**: 
- Check all file paths in the configuration section of each script
- Ensure PDB files are in the correct directory
- Verify `process.csv` exists with correct format
- Make sure pocket PDB files are extracted from ZIP archives

## GPU Support

The `coords_to_graph.py` script supports CUDA acceleration:

```python
# Automatic device selection
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Force CPU usage
device = 'cpu'
```

## Pipeline Statistics

For a typical KIBA dataset with ~228 proteins:

- **Download time**: ~20 minutes (PDB files)
- **Preprocessing**: ~2 minutes
- **Pocket prediction**: ~4 hours (web scraping)
- **Pocket extraction**: ~1 minute
- **Graph conversion**: ~5 minutes (GPU) / ~15 minutes (CPU)

**Total**: ~4.5 hours (mostly pocket prediction)

## Citation

If you use this code in your research, please cite the original tools:

- **AlphaFold**: Jumper et al., "Highly accurate protein structure prediction with AlphaFold," Nature, 2021
- **DoGSite3**: Volkamer et al., "DoGSiteScorer: a web server for automatic binding site prediction," J. Chem. Inf. Model., 2012
- **KIBA Dataset**: Tang et al., "Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets," J. Chem. Inf. Model., 2014

## Related Resources

- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [DoGSite3 Web Server](https://proteins.plus/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [KIBA Dataset](https://www.nature.com/articles/nbt.2914)

## License

This code is provided for research purposes under the MIT License. See `LICENSE` file for details.

Please also check the licenses of:
- AlphaFold structures (CC-BY 4.0)
- DoGSite3 web service (Academic use)
- KIBA dataset

