#!/bin/bash

# Directory structure for Kinetics 400 dataset
root_dl="k400"
annotations_dir="${root_dl}/annotations"

# Create directories if they don't exist
echo "Creating directories..."
[ ! -d $root_dl ] && mkdir -p $root_dl
[ ! -d $annotations_dir ] && mkdir -p $annotations_dir

# CSV file URLs
echo "Preparing to download CSV files..."
csv_files=(
  "https://s3.amazonaws.com/kinetics/400/annotations/train.csv"
  "https://s3.amazonaws.com/kinetics/400/annotations/val.csv"
  "https://s3.amazonaws.com/kinetics/400/annotations/test.csv"
)

# Download each CSV file with progress display
for url in "${csv_files[@]}"; do
  filename=$(basename $url)
  echo "Downloading $filename..."
  curl -C - $url -o "$annotations_dir/$filename" --progress-bar
  echo "✓ Download complete: $filename"
done

echo -e "\nAll CSV files downloaded successfully to $annotations_dir"
echo "You can add these directories to your .gitignore:"
echo "  $root_dl/"
echo "  $annotations_dir/"