# Davis Protein-Drug Binding Pocket Analysis

This repository contains scripts for processing protein structures from the Davis dataset, extracting binding pockets using DoGSite3, and converting them to graph representations for machine learning applications.

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
data_preprocess/Davis/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore patterns
│
├── config_template.py                 # Configuration template
├── example_process.csv                # Example protein ID file
│
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

1. **process.csv** - CSV file containing a column `target_key` with protein IDs

Example `process.csv`:
```csv
target_key
ABL1
EGFR
SRC
...
```

## Usage

### Option 1: Run Complete Pipeline

```bash
python run_pipeline.py --step all
```

This will execute all steps sequentially. Make sure to update the configuration paths in the script.

### Option 2: Run Individual Steps

#### Step 1: Preprocess Protein Data

```bash
python preprocess_alphafold_data.py
```

Update configuration:
- `PROCESS_CSV`: Path to process.csv
- `PDB_PATH`: Directory containing downloaded PDB files

**Output**: `Davis_Protein_Domain_Alphfold3d_dict.pickle`

#### Step 2: Predict Binding Pockets (DoGSite3)

```bash
python scrape_dogsite_pockets.py
```

This script uses Selenium to automate pocket prediction on https://proteins.plus/

**Note**: This step can take several hours. You'll need to manually extract the pocket PDB files from the downloaded ZIP files.

#### Step 3: Extract Pocket Residues

```bash
python extract_pocket_top1.py
python extract_pocket_top2.py
python extract_pocket_top3.py
```

**Outputs**:
- `Davis_protein_domain_pocket_top{1,2,3}index_dict.pickle`
- `Davis_protein_domain_pocket_top{1,2,3}len_dict.pickle`
- `Davis_protein_domain_pocket_top{1,2,3}seq_dict.pickle`

#### Step 4: Convert to Graph Representations

```bash
python coords_to_graph.py
```

**Final Outputs**:
- `Davis_Domain_coord_graph_top1seqid_dict.pickle`
- `Davis_Domain_coord_graph_top2seqid_dict.pickle`
- `Davis_Domain_coord_graph_top3seqid_dict.pickle`

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
PDB_PATH = 'prot_3d_for_Davis'
OUTPUT_FILE = 'output.pickle'
```

You can also use the `config_template.py` file as a reference. Copy it to `config.py` and update the paths:

```bash
cp config_template.py config.py
# Then edit config.py with your paths
```

## License

This code is provided for research purposes under the MIT License. See `LICENSE` file for details.

Please also check the licenses of:
- AlphaFold structures (CC-BY 4.0)
- DoGSite3 web service (Academic use)
- Davis dataset
