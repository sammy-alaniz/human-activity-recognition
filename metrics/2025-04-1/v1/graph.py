import json
import argparse
import os
from collections import Counter
import matplotlib.pyplot as plt

def plot_label_counts(data, output_image_file):
    """
    Counts the correctness labels in the processed data and generates a bar chart.

    Args:
        data (list): The list of dictionaries loaded from the processed JSON file.
        output_image_file (str): Path to save the output bar chart image.
    """
    label_counts = Counter()

    # --- Count the labels ---
    for entry in data:
        queries = entry.get('queries', [])
        if not isinstance(queries, list):
            continue # Skip if queries is not a list

        for query_data in queries:
            # Expecting 4 elements now: [query, response, time, label]
            if isinstance(query_data, list) and len(query_data) == 4:
                label = query_data[3]
                # We only count the relevant labels assigned by the previous script
                if label in ["Correctly Labeled", "Incorrectly Labeled", "Unknown Response"]:
                     label_counts[label] += 1
            # Optional: Add handling or warning for lists not of length 4 if needed

    if not label_counts:
        print("No relevant labels found to plot.")
        return

    # --- Prepare data for plotting ---
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    # --- Create Bar Chart ---
    try:
        fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size if needed
        bars = ax.bar(labels, counts, color=['green', 'red', 'orange', 'grey'][:len(labels)]) # Basic colors

        ax.set_ylabel('Number of Queries')
        ax.set_title('Query Label Correctness Counts')
        ax.set_xticks(range(len(labels))) # Ensure ticks are set correctly
        ax.set_xticklabels(labels, rotation=0) # Rotate if labels overlap

        # Add counts above the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center') # va: vertical alignment

        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # --- Save the plot ---
        plt.savefig(output_image_file)
        print(f"Chart saved successfully to {output_image_file}")
        # plt.show() # Uncomment this if you want to display the plot directly

    except ImportError:
         print("Error: Matplotlib is required for plotting. Please install it (`pip install matplotlib`)")
    except Exception as e:
        print(f"An error occurred during plotting or saving the image: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate a bar chart from labeled JSON video query data.")
    parser.add_argument("input_file", help="Path to the input JSON file (output from the previous script).")
    parser.add_argument("output_image", help="Path to save the output bar chart image (e.g., counts.png).")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Load data from input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
        print(f"Successfully loaded labeled data from {args.input_file}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {args.input_file}: {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading {args.input_file}: {e}")
        return

    # Plot the data
    plot_label_counts(labeled_data, args.output_image)

if __name__ == "__main__":
    main()