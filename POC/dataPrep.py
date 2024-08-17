import os
import shutil
from pathlib import Path
import random
from math import floor

def ensure_dir_exists(dir_path):
    """Ensure the directory exists. If not, create it."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def split_data(source_dir, train_dir, val_dir, test_dir, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    # Ensure the split directories exist
    ensure_dir_exists(train_dir)
    ensure_dir_exists(val_dir)
    ensure_dir_exists(test_dir)
    
    for category in os.listdir(source_dir):
        category_path = Path(source_dir) / category
        if category_path.is_dir():
            files = list(category_path.glob('*'))
            random.shuffle(files)
            train_end = floor(train_prop * len(files))
            val_end = train_end + floor(val_prop * len(files))

            train_files = files[:train_end]
            val_files = files[train_end:val_end]
            test_files = files[val_end:]

            for file in train_files:
                dest_path = Path(train_dir) / category
                ensure_dir_exists(dest_path)
                shutil.copy2(file, dest_path / file.name)
                print(f"Copied {file} to {dest_path / file.name}")
            
            for file in val_files:
                dest_path = Path(val_dir) / category
                ensure_dir_exists(dest_path)
                shutil.copy2(file, dest_path / file.name)
                print(f"Copied {file} to {dest_path / file.name}")
            
            for file in test_files:
                dest_path = Path(test_dir) / category
                ensure_dir_exists(dest_path)
                shutil.copy2(file, dest_path / file.name)
                print(f"Copied {file} to {dest_path / file.name}")

if __name__ == "__main__":
    # Define the paths to the source and destination directories
    source_dir = '/Users/justinhuang/Documents/Developer/ML/CXRML/CXRData_unsplit'
    train_dir = '/Users/justinhuang/Documents/Developer/ML/CXRML/CXRData/train'
    val_dir = '/Users/justinhuang/Documents/Developer/ML/CXRML/CXRData/valid'
    test_dir = '/Users/justinhuang/Documents/Developer/ML/CXRML/CXRData/test'

    # Split the source data into training, validation, and testing sets
    split_data(source_dir, train_dir, val_dir, test_dir, train_prop=0.7, val_prop=0.2, test_prop=0.1)
