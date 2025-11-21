"""
Configuration Template for KIBA Pocket Processing Pipeline

Copy this file to config.py and update the paths according to your system.
"""

# ============================================================================
# Input Data Paths
# ============================================================================

# CSV file containing protein IDs in column 'target_key'
PROCESS_CSV = 'process.csv'

# Text file with protein UniProt IDs (one per line) for downloading
PROTEIN_ID_FILE = 'KIBA_protein_id.txt'

# Directory containing downloaded PDB files from AlphaFold
PDB_PATH = 'prot_3d_for_KIBA'

# ============================================================================
# Pocket PDB File Directories
# ============================================================================

# Directories containing pocket PDB files from DoGSite3
# These should be extracted from the downloaded ZIP files
POCKET_PDB_DIR_TOP1 = 'Domain_Key_Pocket_files_top1'
POCKET_PDB_DIR_TOP2 = 'Domain_Key_Pocket_files_top2'
POCKET_PDB_DIR_TOP3 = 'Domain_Key_Pocket_files_top3'

# ============================================================================
# Download Settings (for Selenium scripts)
# ============================================================================

# Your browser's download folder
DOWNLOAD_FOLDER = r'C:\Users\YOUR_USERNAME\Downloads'

# Target folder for downloaded AlphaFold PDB files
ALPHAFOLD_TARGET_FOLDER = 'Alphafold_pdb_files'

# ============================================================================
# Processing Settings
# ============================================================================

# Maximum sequence length for graph conversion
MAX_SEQ_LEN = 120

# Number of nearest neighbors for k-NN graph
TOP_K = 30

# Device for PyTorch (cuda:0, cuda:1, or cpu)
DEVICE = 'cuda:0'  # Change to 'cpu' if no GPU available

# Number of RBF embeddings
NUM_RBF = 16

# Number of positional embeddings
NUM_POSITIONAL_EMBEDDINGS = 16

# ============================================================================
# Output File Names
# ============================================================================

# Preprocessed protein 3D structure dictionary
PROTEIN_3D_DICT_FILE = 'KIBA_Protein_Domain_Alphfold3d_dict.pickle'

# Pocket index dictionaries (output from extraction scripts)
POCKET_TOP1_INDEX_FILE = 'KIBA_protein_domain_pocket_top1index_dict.pickle'
POCKET_TOP2_INDEX_FILE = 'KIBA_protein_domain_pocket_top2index_dict.pickle'
POCKET_TOP3_INDEX_FILE = 'KIBA_protein_domain_pocket_top3index_dict.pickle'

# Pocket length dictionaries
POCKET_TOP1_LEN_FILE = 'KIBA_protein_domain_pocket_top1len_dict.pickle'
POCKET_TOP2_LEN_FILE = 'KIBA_protein_domain_pocket_top2len_dict.pickle'
POCKET_TOP3_LEN_FILE = 'KIBA_protein_domain_pocket_top3len_dict.pickle'

# Pocket sequence dictionaries
POCKET_TOP1_SEQ_FILE = 'KIBA_protein_domain_pocket_top1seq_dict.pickle'
POCKET_TOP2_SEQ_FILE = 'KIBA_protein_domain_pocket_top2seq_dict.pickle'
POCKET_TOP3_SEQ_FILE = 'KIBA_protein_domain_pocket_top3seq_dict.pickle'

# Intermediate pocket coordinates dictionaries
POCKET_COORDS_TOP1_FILE = 'KIBA_Domain_DoGsite3_top1seqid_dict.pickle'
POCKET_COORDS_TOP2_FILE = 'KIBA_Domain_DoGsite3_top2seqid_dict.pickle'
POCKET_COORDS_TOP3_FILE = 'KIBA_Domain_DoGsite3_top3seqid_dict.pickle'

# Final graph representation dictionaries
GRAPH_TOP1_FILE = 'KIBA_Domain_coord_graph_top1seqid_dict.pickle'
GRAPH_TOP2_FILE = 'KIBA_Domain_coord_graph_top2seqid_dict.pickle'
GRAPH_TOP3_FILE = 'KIBA_Domain_coord_graph_top3seqid_dict.pickle'

# ============================================================================
# Selenium Settings (for web scraping)
# ============================================================================

# Wait times (in seconds)
PAGE_LOAD_WAIT = 5
PANEL_LOAD_WAIT = 2
CALCULATION_WAIT = 30
DOWNLOAD_WAIT = 5

# DoGSite3 URL
DOGSITE_URL = 'https://proteins.plus/'
