#!/bin/bash

# current folder
cwd=$(pwd)

# src_folder
target="orbit_benchmark_224_better_smaller"
new="orbit_benchmark_224"
src=${cwd/$target/$new}
# echo "$src"

# Check if the target directory exists, if not, create it
if [ ! -d "$target" ]; then
    echo "Target directory does not exist. Creating it..."
    mkdir -p "$target"
fi

# Find all files in the source directory, loop through them to create symlinks in the target directory
find "$src" -type f | while read -r file; do

	if [[ "$file" == *"clean"* ]]; then
		# echo "Skipping"
		continue
	fi

    # Replace the part of the path that differs
	# echo "$file"
    symlink_path="${file/$new/$target}"
    
    # Get the directory of the symlink to be created, and make sure it exists
    symlink_dir=$(dirname "$symlink_path")
    if [ ! -d "$symlink_dir" ]; then
        mkdir -p "$symlink_dir"
    fi
    
    # Create the symlink
    ln -s "$file" "$symlink_path"
done

echo "All symlinks have been created successfully."

