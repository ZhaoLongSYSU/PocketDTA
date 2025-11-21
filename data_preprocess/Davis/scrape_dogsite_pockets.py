"""
DoGSite3 Protein Pocket Prediction Scraper for Davis Dataset

This script uses Selenium to automate the process of submitting protein structures
to the DoGSite3 web server (https://proteins.plus/) and downloading the predicted
binding pocket files.

Requirements:
    - Selenium WebDriver
    - Chrome browser and chromedriver
    - PDB files already downloaded

Note: This process can take several hours depending on the number of proteins.
"""

import pandas as pd
from tqdm import tqdm
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Configuration - Update these paths according to your setup
PROCESS_CSV = 'process.csv'  # CSV file containing target protein IDs
PDB_PATH = 'prot_3d_for_Davis'  # Directory containing PDB files
DOWNLOAD_DIR = r"C:\Users\YOUR_USERNAME\Downloads"  # Browser download directory

# Read protein IDs
df = pd.read_csv(PROCESS_CSV)
target_key = list(df['target_key'].unique())

# Selenium configuration
options = webdriver.ChromeOptions()
prefs = {'download.default_directory': DOWNLOAD_DIR}
options.add_experimental_option('prefs', prefs)

driver = webdriver.Chrome(options=options)

# Track errors
proteins_id_error = []


def pocket_prediction(pro_id, pdb_file):
    """
    Submit protein to DoGSite3 and download predicted pockets.
    
    Args:
        pro_id: Protein ID
        pdb_file: Path to PDB file
    """
    try:
        # Open DoGSite3 website
        driver.get("https://proteins.plus/")

        # Find input box and enter protein PDB file path
        input_box = driver.find_element(By.CSS_SELECTOR, "#pdb_file_pathvar")
        input_box.send_keys(pdb_file)

        # Click GO button
        go_button = driver.find_element(By.NAME, "commit")
        go_button.click()

        # Wait for page to load
        time.sleep(5)

        # Click "DoGSite3 Binding site detection" panel to expand
        dogsite_panel_heading = driver.find_element(By.ID, "headingDoGSite3")
        dogsite_panel_heading.click()

        # Wait for panel content to load
        time.sleep(2)

        # Click "DoGSite3" button using XPath
        dogsite3_button_xpath = "//div[@id='headingDoGSite3']/following-sibling::div//a[@id='changeViewTab']"
        dogsite3_button = driver.find_element(By.XPATH, dogsite3_button_xpath)
        dogsite3_button.click()
        time.sleep(2)

        # Click "Calculate" button
        calculate_button_xpath = "//div[@id='dogsite3result']//input[@type='submit'][@value='Calculate']"
        calculate_button = driver.find_element(By.XPATH, calculate_button_xpath)
        calculate_button.click()
        time.sleep(30)  # Wait for calculation to complete

        # Click download button
        download_button = driver.find_element(
            By.CSS_SELECTOR, 
            '#dogsite3result > div.row-fluid > div:nth-child(1) > form > input.btn-primary'
        )
        download_button.click()

        # Wait for download to complete
        time.sleep(5)
        
        # Rename downloaded file
        list_of_files = os.listdir(DOWNLOAD_DIR)
        full_path = [os.path.join(DOWNLOAD_DIR, i) for i in list_of_files]
        latest_file = max(full_path, key=os.path.getctime)
        
        # If the latest downloaded file is a zip file, rename it
        if latest_file.endswith(".zip"):
            new_name = os.path.join(DOWNLOAD_DIR, f"{pro_id}.zip")
            if os.path.exists(new_name):
                os.remove(new_name)  # Remove if already exists
            os.rename(latest_file, new_name)

        print(f"{pro_id} completed!")
        
    except Exception as e:
        print(f"Error downloading {pro_id}: {str(e)}")
        proteins_id_error.append(pro_id)


# Main processing loop
if __name__ == "__main__":
    for pro_id in tqdm(target_key, desc="Scraping pockets"):
        pdb_file = os.path.join(PDB_PATH, f'{pro_id}.pdb')
        
        if not os.path.exists(pdb_file):
            print(f"Warning: PDB file not found for {pro_id}")
            proteins_id_error.append(pro_id)
            continue
            
        pocket_prediction(pro_id, pdb_file)

    # Quit browser
    driver.quit()

    # Print failed downloads
    if proteins_id_error:
        print("\nFailed to process the following proteins:")
        for error_id in proteins_id_error:
            print(error_id)
    else:
        print("\nAll proteins processed successfully!")
