import os
import sys
from pathlib import Path
from tqdm import tqdm

def create_symlinks(image_list_path, src_base, target_base):
    with open(image_list_path, 'r') as file:
        paths = [line.strip() for line in file]
    
    for path in tqdm(paths, desc="Creating symlinks"):
        # print(path)
        full_src_path = src_base / Path(str("./" + path))
        full_target_path = target_base / Path(str("./" + path))

        # print(full_src_path)

        # print(src_base)
        # print(path)
        
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
    src_base = Path("/home/zhumakhanova/Desktop/cv703_project/")
    target_base = Path("/home/zhumakhanova/Desktop/cv703_project/orbit_benchmark_224/dataset_only_good")
    
    create_symlinks(image_list_path, src_base, target_base)
