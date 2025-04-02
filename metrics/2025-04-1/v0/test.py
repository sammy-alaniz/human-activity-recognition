import json
import argparse
import os

def analyze_video_queries(data):
    """
    Analyzes query responses in the video data, checks against the activity label,
    and adds a correctness label.

    Args:
        data (list): A list of dictionaries loaded from the JSON file.

    Returns:
        list: The modified list of dictionaries with added correctness labels.
    """
    if not isinstance(data, list):
        raise ValueError("Input data must be a list of dictionaries.")

    for entry in data:
        if not isinstance(entry, dict):
            print(f"Warning: Skipping non-dictionary item in list: {entry}")
            continue # Skip if the item is not a dictionary

        activity = entry.get('activity', '').lower() # Get activity label, handle missing key
        queries = entry.get('queries', []) # Get queries list, handle missing key

        if not isinstance(queries, list):
             print(f"Warning: Skipping entry with non-list 'queries' field for activity '{activity}': {entry.get('filename', 'N/A')}")
             continue # Skip if queries is not a list

        for query_data in queries:
            # Ensure query_data is a list and has at least 2 elements (query, response)
            if not isinstance(query_data, list) or len(query_data) < 2:
                print(f"Warning: Skipping malformed query data in entry for activity '{activity}': {entry.get('filename', 'N/A')}")
                continue # Skip malformed query

            query_str = str(query_data[0]).lower() # Ensure string and lower case
            response_str = str(query_data[1]).strip().lower() # Ensure string, strip whitespace, lower case

            extracted_label = None
            if response_str.startswith("yes"):
                extracted_label = "Yes"
            elif response_str.startswith("no"):
                extracted_label = "No"

            expected_label = None
            correctness_label = "N/A" # Default label

            # --- Determine Expected Label based on Query and Activity ---
            if "cooking" in query_str:
                expected_label = "Yes" if "cooking" in activity else "No"
            elif "cleaning" in query_str:
                 # Handle cases like 'washing_dishes' which imply cleaning
                is_cleaning_activity = "cleaning" in activity or "washing" in activity
                expected_label = "Yes" if is_cleaning_activity else "No"
            # Add more rules here if needed for other query types

            # --- Compare and Assign Correctness Label ---
            if expected_label is not None: # Only evaluate if we could determine an expected label
                if extracted_label is None:
                    correctness_label = "Unknown Response"
                elif extracted_label == expected_label:
                    correctness_label = "Correctly Labeled"
                else:
                    correctness_label = "Incorrectly Labeled"

            # --- Append the label ---
            # Check if label already exists (e.g., from previous run)
            if len(query_data) == 3:
                 query_data.append(correctness_label)
            elif len(query_data) == 4:
                 query_data[3] = correctness_label # Overwrite if exists
            else:
                 # Handle unexpected list length if necessary, or just append
                 query_data.append(correctness_label)

    return data

def main():
    parser = argparse.ArgumentParser(description="Analyze JSON video query data and label correctness.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to save the output JSON file.")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Load data from input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        print(f"Successfully loaded data from {args.input_file}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {args.input_file}: {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading {args.input_file}: {e}")
        return

    # Process the data
    try:
        processed_data = analyze_video_queries(video_data)
        print("Data analysis complete.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return

    # Save data to output file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4) # Use indent for readability
        print(f"Successfully saved processed data to {args.output_file}")
    except Exception as e:
        print(f"An error occurred while saving to {args.output_file}: {e}")

if __name__ == "__main__":
    main()