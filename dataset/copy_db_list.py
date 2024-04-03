import os
import sys
from pathlib import Path
from tqdm import tqdm

def create_symlinks(image_list_path, src_base, target_base):
    with open(image_list_path, 'r') as file:
        paths = [line.strip() for line in file if 'clean' not in line]
    
    for path in tqdm(paths, desc="Creating symlinks"):
        full_src_path = src_base / path
        full_target_path = target_base / path
        
        # Ensure the target directory exists
        full_target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the symlink, if the source file exists
        if full_src_path.exists():
            if not full_target_path.exists():
                os.symlink(full_src_path, full_target_path)
        else:
            print(f"Warning: Source file does not exist - {full_src_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_symlinks.py /path/to/your/imageList.txt")
        sys.exit(1)
    
    image_list_path = sys.argv[1]
    src_base = Path("/home/sebastian.cavada/Documents/scsv/semester2/CV703/datasets/orbit_benchmark_224")
    target_base = Path("/home/sebastian.cavada/Documents/scsv/semester2/CV703/datasets/orbit_dataset_224_better_smaller")
    
    create_symlinks(image_list_path, src_base, target_base)
