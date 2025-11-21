<div align="center">

# PocketDTA: An Advanced Multimodal Architecture for Enhanced Prediction of Drug-Target Affinity from 3D Structural Data of Target Binding Pockets

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![rdkit](https://img.shields.io/badge/-rdkit_2023.3.2+-792ee5?logo=rdkit&logoColor=white)](https://anaconda.org/conda-forge/rdkit/)
[![torch-geometric](https://img.shields.io/badge/torch--geometric-2.3.1+-792ee5?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![deepchem](https://img.shields.io/badge/deepchem-2.7.1+-792ee5?logo=deepchem&logoColor=white)](https://deepchem.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Preprocessing](#-data-preprocessing)
- [Dataset](#-dataset)
- [Pre-trained Models](#-pre-trained-models)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Interpretability Analysis](#-interpretability-analysis)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)

---

## 📌 Introduction 

### Motivation
Accurately predicting drug-target binding affinity (DTA) is crucial for drug discovery and repurposing. Although deep learning has been widely used in this field, it still faces three major challenges:

1. **Insufficient generalization performance** across different protein families and chemical spaces
2. **Inadequate use of 3D structural information** from both proteins and small molecules
3. **Poor interpretability** limiting the understanding of binding mechanisms

### Our Solution
To address these challenges, we developed **PocketDTA**, an advanced multimodal deep learning architecture that:

- ✅ **Enhances generalization** through pre-trained models (ESM-2 for proteins, GraphMVP for molecules)
- ✅ **Leverages 3D structural data** from protein binding pockets and drug conformations
- ✅ **Improves interpretability** via bilinear attention networks to identify key interactions
- ✅ **Processes multiple binding sites** by handling top-3 predicted binding pockets
- ✅ **Achieves SOTA performance** on benchmark datasets (Davis and KIBA)

### Results Highlights
- 🏆 **Outperforms existing methods** in comparative analysis on optimized datasets
- 🔬 **Validated interpretability** through molecular docking and literature confirmation
- 💪 **Robust generalization** demonstrated in cold-start experiments
- 🎯 **Identifies key interactions** between drug functional groups and amino acid residues

---

## ✨ Key Features

- **🧬 Advanced Protein Encoding**: Utilizes ESM-2 protein language model for rich sequence representations
- **💊 3D Molecular Representations**: Employs GraphMVP for learning from 3D molecular structures
- **🔍 Multi-Pocket Analysis**: Processes top-3 binding pockets for comprehensive protein-drug interactions
- **🧠 GVP-GNN Architecture**: Custom Geometric Vector Perceptron Graph Neural Networks for 3D geometry
- **🎨 Bilinear Attention**: Captures cross-modal interactions between proteins and molecules
- **📊 High Interpretability**: Provides attention weights for understanding binding mechanisms

---

## 🚀 Architecture

![PocketDTA Architecture](https://github.com/zhaolongNCU/PocketDTA/blob/main/PocketDTA.jpg)

The PocketDTA architecture consists of:

1. **Protein Branch**:
   - ESM-2 sequence encoder
   - GVP-GNN for 3D binding pocket structures (top-3 pockets)
   - Multi-head attention for pocket aggregation

2. **Molecule Branch**:
   - GraphMVP encoder for 3D molecular conformations
   - Graph neural network for molecular feature extraction

3. **Interaction Module**:
   - Bilinear attention network
   - Cross-modal fusion layer
   - Affinity prediction head

---

## 💻 Installation

### Prerequisites

- **Operating System**: Linux (recommended), Windows, or macOS
- **Python**: 3.7.16
- **CUDA**: 11.7 (for GPU support)
- **RAM**: 24GB+ recommended for training

### Step 1: Clone the Repository

```bash
git clone https://github.com/zhaolongNCU/PocketDTA.git
cd PocketDTA
```

### Step 2: Create Conda Environment

```bash
# Create a new conda environment
conda create -n PocketDTA python=3.7
conda activate PocketDTA
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==1.13.0+cu117 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric and dependencies
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Install other dependencies
pip install -r requirements.txt
```

### Alternative: One-Command Installation

```bash
pip install -r requirements.txt
```

**Note**: The `requirements.txt` contains all necessary packages with specified versions.

---

## 🚀 Quick Start

### 1. Download Pre-trained Models

Download the required pre-trained model weights:

- **ESM-2** (650M parameters): [Download Link](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)
- **GraphMVP**: Already included in `dataset/Davis/` and `dataset/KIBA/`

Place ESM-2 model file in the appropriate directory (see [Pre-trained Models](#-pre-trained-models)).

### 2. Download Dataset Files

Download 3D structure files from [Google Cloud Drive](https://drive.google.com/drive/folders/1qJXsxkTSgwPSTpu-XmIUh2rD2jJ1KuGQ):

- Target 3D structures (`.pdb` files)
- Top-3 binding pocket files (`.pdb` files)

Place them in the corresponding dataset folders.

### 3. Train the Model

```bash
# Train on Davis dataset with seed 0
python main.py --task Davis --r 0

# Train on KIBA dataset
python main.py --task KIBA --r 0
```

### 4. Evaluate Performance

Results will be saved in the `result/` directory, including:
- Training/validation metrics
- Model checkpoints
- Prediction results

---

## 🔧 Data Preprocessing

We provide complete data preprocessing pipelines for preparing custom datasets or reproducing our results from scratch.

### Pipeline Overview

The `data_preprocess/` directory contains standalone preprocessing tools for both Davis and KIBA datasets:

```
data_preprocess/
├── Davis/              # Davis dataset preprocessing pipeline
│   ├── README.md      # Complete documentation and usage guide
│   └── ...            # Preprocessing scripts
│
├── KIBA/              # KIBA dataset preprocessing pipeline
│   ├── README.md      # Complete documentation and usage guide
│   └── ...            # Preprocessing scripts
│
└── requirements.txt   # Preprocessing dependencies
```

### Quick Start

```bash
# Navigate to dataset-specific directory
cd data_preprocess/Davis  # or KIBA

# Run automated pipeline
python run_pipeline.py --step all
```

### Workflow

The pipeline processes: **AlphaFold PDB files** → **DoGSite3 pockets** → **Graph representations**

Final outputs: `{Dataset}_Domain_coord_graph_top{1,2,3}seqid_dict.pickle`

📖 **For detailed instructions, custom dataset processing, and troubleshooting, see:**
- [`data_preprocess/Davis/README.md`](data_preprocess/Davis/README.md)
- [`data_preprocess/KIBA/README.md`](data_preprocess/KIBA/README.md)

---

### Benchmark Datasets

We provide two benchmark datasets for drug-target affinity prediction:

| Dataset | Proteins | Compounds | Interactions | Affinity Range | Source |
|---------|----------|-----------|--------------|----------------|--------|
| **Davis** | 442 | 68 | 30,056 | Kd values | Kinase inhibitors |
| **KIBA** | 229 | 2,111 | 118,254 | KIBA scores | Bioactivity database |

### Dataset Structure

```
dataset/
├── Davis/
│   ├── process.csv                          # Protein-compound pairs
│   ├── GraphMVP.pth                         # Pre-trained GraphMVP model
│   ├── Davis_Domain_coord_graph_top1seqid_dict.pickle
│   ├── Davis_Domain_coord_graph_top2seqid_dict.pickle
│   └── Davis_Domain_coord_graph_top3seqid_dict.pickle
│
└── KIBA/
    └── (same structure as Davis/)
```

### Required Downloads

Download from [Google Cloud Drive](https://drive.google.com/drive/folders/1qJXsxkTSgwPSTpu-XmIUh2rD2jJ1KuGQ):

- **Protein 3D structures** (`.pdb` files from AlphaFold)
- **Binding pocket files** (top-3 predicted pockets from DoGSite3)

### Data Format

Each `.pickle` file contains a dictionary:
```python
{
    'PROTEIN_ID': torch_geometric.data.Data(
        x=...,          # Coordinates
        seq=...,        # Sequence
        node_s=...,     # Scalar features
        node_v=...,     # Vector features
        edge_s=...,     # Edge scalar features
        edge_v=...,     # Edge vector features
        edge_index=..., # Graph structure
    ),
    ...
}
```

---

## 🎯 Pre-trained Models

PocketDTA leverages several pre-trained models for enhanced performance:

### Required Models

| Model | Purpose | Size | Download Link |
|-------|---------|------|---------------|
| **ESM-2** | Protein sequence encoding | 650M params | [Download](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt) |
| **GraphMVP** | Molecular graph pre-training | Included | In `dataset/` folders |

### Optional Models (for ablation studies)

| Model | Purpose | Download Link |
|-------|---------|---------------|
| **ProtBert** | Alternative protein encoder | [Zenodo](https://zenodo.org/records/4633691) |
| **ProtT5** | Alternative protein encoder | [Zenodo](https://zenodo.org/records/4644188) |
| **3DInfomax** | 3D molecular pre-training | [GitHub](https://github.com/HannesStark/3DInfomax) |

### Model Placement

Place downloaded models in the appropriate directories as specified in the configuration files in `configs/`.

---

## 🎓 Training

### Basic Training

Train on Davis dataset:
```bash
python main.py --task Davis --r 0
```

Train on KIBA dataset:
```bash
python main.py --task KIBA --r 0
```

### Multi-Seed Training

Run experiments with multiple random seeds (Linux):
```bash
./training.sh
```

This script trains models with seeds 0-4 for robust performance evaluation.

### Training Parameters

Key command-line arguments:

- `--task`: Dataset to use (`Davis` or `KIBA`)
- `--r`: Random seed for reproducibility
- `--epochs`: Number of training epochs (default: 1000)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.0001)
- `--device`: GPU device (default: cuda:0)

### Monitoring Training

Training logs and checkpoints are saved in `result/{task}/`:
- Training/validation losses
- Performance metrics (MSE, CI, R², etc.)
- Model checkpoints
- Prediction results

---

## 📊 Evaluation

### Ablation Studies

**Representation Ablation** - Test different pre-trained encoders:
```bash
./Ablation.sh
```

This compares:
- ESM-2 vs ProtBert vs ProtT5 (protein encoders)
- GraphMVP vs 3DInfomax (molecular encoders)

**Module Ablation** - Test model components:
```bash
./Ablation_module.sh
```

This evaluates:
- Impact of multi-pocket processing
- Effect of bilinear attention
- Contribution of 3D geometric features

### Cold-Start Experiments

Test generalization to unseen proteins/compounds:
```bash
./Cold.sh
```

Three scenarios:
- **Cold-start proteins**: Unseen target proteins
- **Cold-start compounds**: Unseen drug molecules
- **Cold-start pairs**: Both protein and compound unseen

---

## 🔬 Interpretability Analysis

Identify key molecular interactions using attention weights:

```bash
python interaction_weight.py --task Davis --model DTAPredictor_test --r 2 --use-test True
```

### Outputs

The script generates:
- **Atomic attention weights**: Important functional groups in the drug
- **Residue attention weights**: Key amino acids in the binding pocket
- **Visualization files**: Heatmaps of protein-drug interactions

### Analysis Workflow

1. Select protein-compound pair for analysis
2. Run `interaction_weight.py` to compute attention weights
3. Visualize interactions using molecular docking software
4. Validate findings with literature and experimental data

### Use Cases

- **Drug design**: Identify modifications to improve binding
- **Lead optimization**: Focus on key interaction sites
- **Mechanism understanding**: Explain binding affinity predictions

---

## 📁 Project Structure

```
PocketDTA/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── PocketDTA.jpg               # Architecture diagram
│
├── data_preprocess/            # Data preprocessing pipelines
│   ├── Davis/                  # Davis dataset processing
│   ├── KIBA/                   # KIBA dataset processing
│   └── requirements.txt        # Preprocessing dependencies
│
├── dataset/                    # Processed datasets
│   ├── Davis/                  # Davis dataset files
│   └── KIBA/                   # KIBA dataset files
│
├── configs/                    # Configuration files
│
├── src/                        # Source code
│   ├── model.py               # Main PocketDTA model
│   ├── gvp_gnn.py             # GVP-GNN implementation
│   ├── compound_gnn_model.py  # Molecular encoder
│   ├── protein_to_graph.py    # Protein graph construction
│   ├── data.py                # Data loading utilities
│   └── featurizers/           # Feature extraction modules
│
├── main.py                     # Training script
├── train_test.py              # Training/testing functions
├── interaction_weight.py       # Interpretability analysis
├── utils_dta.py               # Utility functions
│
├── Radam.py                    # RAdam optimizer
├── lookahead.py               # Lookahead optimizer
│
├── training.sh                 # Multi-seed training
├── Ablation.sh                # Representation ablation
├── Ablation_module.sh         # Module ablation
└── Cold.sh                    # Cold-start experiments
```

---

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{10.1093/bioinformatics/btae594,
    author = {Zhao, Long and Wang, Hongmei and Shi, Shaoping},
    title = "{PocketDTA: An advanced multimodal architecture for enhanced prediction 
             of drug-target affinity from 3D structural data of target binding pockets}",
    journal = {Bioinformatics},
    pages = {btae594},
    year = {2024},
    month = {10},
    doi = {10.1093/bioinformatics/btae594},
    url = {https://doi.org/10.1093/bioinformatics/btae594},
}
```

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

Please also check licenses for:
- **AlphaFold structures**: CC-BY 4.0
- **DoGSite3**: Academic use only
- **Pre-trained models**: Check individual model licenses

---

## 🔗 Related Resources

- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [DoGSite3 Web Server](https://proteins.plus/)
- [ESM-2 Protein Language Model](https://github.com/facebookresearch/esm)
- [GraphMVP](https://github.com/chao1224/GraphMVP)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Davis Dataset](https://www.nature.com/articles/nbt.1990)
- [KIBA Dataset](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z)

---

<div align="center">


⭐ If you find this project helpful, please consider giving it a star!

</div>
