#!/bin/bash

src="/home/sebastian.cavada/Documents/scsv/semester2/CV703/datasets/orbit_benchmark_224"
target="/home/sebastian.cavada/Documents/scsv/semester2/CV703/datasets/orbit_benchmark_224_better_smaller"
imageList="./IFres_v4.txt" # <-- Replace with the actual path to your list

# Check if the target directory exists, if not, create it
if [ ! -d "$target" ]; then
	echo "Target directory does not exist. Creating it..."
	mkdir -p "$target"
fi

# Process each line in the image list file
while IFS= read -r line; do
	fullSrcPath="$line"
	echo "fsrc: $line"

	baseDir=$(dirname "$line")
	echo "basedir $baseDir"
	targetDir="${baseDir/orbit_benchmark_224/orbit_benchmark_224_better_smaller}"

	echo "targetdir $targetDir"

	# Ensure the target directory exiss
	if [ ! -d "$targetDir" ]; then
		echo "Creating directory: $targetDir"
	        mkdir -p "$targetDir"
        fi

	# Create the symlink, if the source file exists
	if [ -f "$fullSrcPath" ]; then
		echo "link $targetDir/$(basename "$line")"
		ln -s "$fullSrcPath" "$targetDir/$(basename "$line")"
	else
	        echo "Warning: Source file does not exist - $fullSrcPath"
	fi
done < "$imageList"

echo "Symlink creation complete. Hopefully, this helps you organize better."

