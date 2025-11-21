"""
AlphaFold PDB File Downloader

This script downloads PDB files from the AlphaFold database using Selenium.
It reads protein IDs from a text file and downloads the corresponding PDB structures.

Requirements:
    - Selenium WebDriver
    - Chrome browser
    - chromedriver
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import shutil

# Configuration - Update these paths according to your setup
PROTEIN_ID_FILE = "KIBA_protein_id.txt"  # File containing protein IDs (one per line)
TARGET_FOLDER = "Alphafold_pdb_files"  # Folder to save downloaded PDB files
DOWNLOAD_FOLDER = r"D:/Backup/Downloads"  # Your browser's download folder

# Create target folder if it doesn't exist
os.makedirs(TARGET_FOLDER, exist_ok=True)

# Read protein IDs from file
identifiers = []
with open(PROTEIN_ID_FILE, "r") as file:
    for line in file:
        protein_id = line.strip()  # Remove trailing whitespace and newline
        identifiers.append(protein_id)

# Track failed downloads
error_identifiers = []

# Initialize Chrome WebDriver
driver = webdriver.Chrome()

def download_pdb_file(identifier):
    """
    Download PDB file for a given protein identifier from AlphaFold.
    
    Args:
        identifier: Protein UniProt ID
    """
    url = f"https://alphafold.com/entry/{identifier}"

    try:
        driver.get(url)
        time.sleep(5)  # Wait for page to load

        # Find and click PDB file download button
        pdb_button = driver.find_element(By.LINK_TEXT, value="PDB file")
        pdb_button.click()

        # Wait for download to complete
        time.sleep(5)

        # Move downloaded file to target folder
        downloaded_file = os.path.join(DOWNLOAD_FOLDER, f"AF-{identifier}-F1-model_v4.pdb")
        target_path = os.path.join(TARGET_FOLDER, f"AF-{identifier}-F1-model_v4.pdb")
        shutil.move(downloaded_file, target_path)
        print(f"Downloaded and moved file to {target_path}")

    except Exception as e:
        print(f"Error downloading {identifier}: {str(e)}")
        error_identifiers.append(identifier)

# Download all proteins
for protein_id in identifiers:
    download_pdb_file(protein_id)
    print(f"{protein_id} completed!")

# Close browser
driver.quit()

# Print failed downloads
if error_identifiers:
    print("\nFailed to download the following identifiers:")
    for error_id in error_identifiers:
        print(error_id)
else:
    print("\nAll downloads completed successfully!")
