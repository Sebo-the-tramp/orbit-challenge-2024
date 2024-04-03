import os
import sys
from pathlib import Path
from tqdm import tqdm

def create_symlinks(src_base, target_base, method, constraint):
    print("Reading all the immense amount of files")
    # List all files, excluding those containing 'clean'
    # files = [path for path in src_base.rglob('*') if 'clean' not in str(path) and path.is_file()]
    files = [path for path in src_base.rglob('*') if constraint not in str(path) and path.is_file()]
    print("done")
    
    for file_path in tqdm(files, desc="Creating symlinks for " + method + " - " + constraint):
        # Construct the relative path from the source base to use in the target
        relative_path = file_path.relative_to(src_base)
        target_path = target_base / relative_path
        
        # Ensure the target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink, checking if it already exists first
        if not target_path.exists():
            os.symlink(file_path, target_path)

if __name__ == "__main__":
    src_base = Path("/home/sebastian.cavada/Documents/scsv/semester2/CV703/datasets/orbit_benchmark_224")
    target_base = Path("/home/sebastian.cavada/Documents/scsv/semester2/CV703/datasets/orbit_dataset_224_better_smaller")

    # Ensure the target base exists
    target_base.mkdir(parents=True, exist_ok=True)

    types = [('test', 'clean'), ('train', 'clean'), ('train', 'cluttered'), ('val', 'clean'), ('val', 'cluttered')]

    for method, constraint in types:
        src = src_base / method
        target = target_base / method
        create_symlinks(src, target, method, constraint)