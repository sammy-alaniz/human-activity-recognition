#!/usr/bin/env python3
import pandas as pd
import sys
import os

def get_unique_labels_pandas(csv_path):
    """Extract all unique labels from the Kinetics 400 CSV file using pandas."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique labels and sort
    unique_labels = sorted(df['label'].unique())
    
    # Count occurrences
    label_counts = df['label'].value_counts().to_dict()
    
    return unique_labels, label_counts

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_labels_pandas.py <path_to_csv_file>")
        print("Example: python get_labels_pandas.py k400/annotations/train.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)
        
    labels, counts = get_unique_labels_pandas(csv_path)
    
    print(f"Found {len(labels)} unique labels in {csv_path}")
    print("\nUnique labels:")
    for label in labels:
        print(f"- {label} ({counts[label]} occurrences)")
        
    # Optionally save to a file
    output_file = "unique_labels.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"\nLabels saved to {output_file}")