"""
Complete KIBA Pocket Processing Pipeline

This script runs the complete workflow from downloading PDB files to generating
graph representations of protein binding pockets.

Usage:
    python run_pipeline.py --step all
    python run_pipeline.py --step preprocess
    python run_pipeline.py --step extract
    python run_pipeline.py --step graph

Note: The download and scraping steps require manual intervention and are not
included in the automatic pipeline.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


class PipelineRunner:
    """Manages the execution of the KIBA pocket processing pipeline."""
    
    def __init__(self, config_file=None):
        """
        Initialize pipeline runner.
        
        Args:
            config_file: Optional configuration file path
        """
        self.base_dir = Path(__file__).parent
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file):
        """Load configuration from file or use defaults."""
        default_config = {
            'process_csv': 'process.csv',
            'pdb_path': 'prot_3d_for_KIBA',
            'pocket_top1_dir': 'Domain_Key_Pocket_files_top1',
            'pocket_top2_dir': 'Domain_Key_Pocket_files_top2',
            'pocket_top3_dir': 'Domain_Key_Pocket_files_top3',
            'device': 'cuda:0',  # or 'cpu'
        }
        
        # TODO: Load from config file if provided
        return default_config
    
    def run_step(self, script_name, description):
        """
        Run a single pipeline step.
        
        Args:
            script_name: Name of the Python script to run
            description: Human-readable description of the step
        """
        print(f"\n{'='*70}")
        print(f"Step: {description}")
        print(f"Script: {script_name}")
        print(f"{'='*70}\n")
        
        script_path = self.base_dir / script_name
        if not script_path.exists():
            print(f"Error: Script not found: {script_path}")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.base_dir,
                check=True,
                capture_output=False
            )
            print(f"\n✓ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {description} failed with error code {e.returncode}")
            return False
    
    def step_preprocess(self):
        """Run preprocessing step."""
        return self.run_step(
            'preprocess_alphafold_data.py',
            'Preprocess AlphaFold PDB Data'
        )
    
    def step_extract_pockets(self):
        """Run pocket extraction steps."""
        steps = [
            ('extract_pocket_top1.py', 'Extract Top 1 Pocket Residues'),
            ('extract_pocket_top2.py', 'Extract Top 2 Pocket Residues'),
            ('extract_pocket_top3.py', 'Extract Top 3 Pocket Residues'),
        ]
        
        for script, description in steps:
            if not self.run_step(script, description):
                return False
        return True
    
    def step_convert_to_graphs(self):
        """Run graph conversion step."""
        return self.run_step(
            'coords_to_graph.py',
            'Convert Coordinates to Graph Representations'
        )
    
    def run_all(self):
        """Run the complete pipeline (excluding download and scraping)."""
        print("\n" + "="*70)
        print("KIBA Pocket Processing Pipeline")
        print("="*70)
        
        steps = [
            ('preprocess', self.step_preprocess),
            ('extract', self.step_extract_pockets),
            ('graph', self.step_convert_to_graphs),
        ]
        
        for step_name, step_func in steps:
            print(f"\nRunning: {step_name}")
            if not step_func():
                print(f"\nPipeline stopped at step: {step_name}")
                return False
        
        print("\n" + "="*70)
        print("Pipeline completed successfully!")
        print("="*70)
        print("\nGenerated files:")
        print("  - KIBA_Protein_Domain_Alphfold3d_dict.pickle")
        print("  - KIBA_Domain_coord_graph_top1seqid_dict.pickle")
        print("  - KIBA_Domain_coord_graph_top2seqid_dict.pickle")
        print("  - KIBA_Domain_coord_graph_top3seqid_dict.pickle")
        print("\nThese files are ready for use in machine learning models!")
        
        return True
    
    def check_prerequisites(self):
        """Check if required input files exist."""
        print("Checking prerequisites...")
        
        required_files = [
            self.config['process_csv'],
        ]
        
        required_dirs = [
            self.config['pdb_path'],
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(f"File: {file}")
        
        for dir in required_dirs:
            if not os.path.exists(dir):
                missing.append(f"Directory: {dir}")
        
        if missing:
            print("\n⚠ Missing required files/directories:")
            for item in missing:
                print(f"  - {item}")
            return False
        
        print("✓ All prerequisites found")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run the KIBA pocket processing pipeline'
    )
    parser.add_argument(
        '--step',
        choices=['all', 'preprocess', 'extract', 'graph', 'check'],
        default='all',
        help='Which step to run (default: all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PipelineRunner(config_file=args.config)
    
    # Check prerequisites
    if args.step == 'check':
        pipeline.check_prerequisites()
        return
    
    if not pipeline.check_prerequisites():
        print("\nPlease ensure all required files are in place before running.")
        print("See README.md for details.")
        return
    
    # Run requested step(s)
    if args.step == 'all':
        pipeline.run_all()
    elif args.step == 'preprocess':
        pipeline.step_preprocess()
    elif args.step == 'extract':
        pipeline.step_extract_pockets()
    elif args.step == 'graph':
        pipeline.step_convert_to_graphs()


if __name__ == "__main__":
    main()
