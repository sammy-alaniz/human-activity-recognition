import json
import argparse
import os
from collections import defaultdict
import re
import csv # Import the csv module

# --- Configuration: Define Target Queries and Action Mapping ---
# List of exact queries to evaluate (excluding non-Yes/No ones like "Describe...")
TARGET_QUERIES = {
    # Query String : Action Keyword(s)
    'Does it look like someone is cleaning?': 'cleaning',
    'Does it look like someone is cooking chicken': 'cooking_chicken', # Be specific
    'Does it look like someone is reading book': 'reading_book',
    'Does it look like someone is cleaning shoes': 'cleaning_shoes',
    'Does it look like someone is mowing lawn': 'mowing_lawn',
    'Does it look like someone is carrying baby': 'carrying_baby',
    'Does it look like someone is sanding floor': 'sanding_floor',
    'Does it look like someone is petting animal': 'petting_animal',
    'Does it look like someone is cleaning floor': 'cleaning_floor',
    'Does it look like someone is using computer': 'using_computer',
    'Does it look like someone is folding clothes': 'folding_clothes',
    'Does it look like someone is flipping pancake': 'flipping_pancake', # Specific cooking
    'Does it look like someone is cleaning windows': 'cleaning_windows',
    'Does it look like someone is decorating the christmas tree': 'decorating_christmas_tree',
    'Does it look like someone is washing dishes': 'washing_dishes', # Specific cleaning
}
# -------------------------------------------------------------

def determine_ground_truth(activity_label, query_action_key):
    """
    Determines the expected ground truth ('Yes' or 'No') based on the
    activity label and the specific action key derived from the query.
    (Refine keywords here based on your specific activity labels)
    """
    activity_label_lower = activity_label.lower()
    query_action_lower = query_action_key.lower()

    action_keywords = {
        'cleaning': ['clean', 'wash', 'tidy', 'sweep', 'mop', 'scrub', 'dust'],
        'cooking_chicken': ['cook', 'chicken', 'meal', 'food', 'kitchen', 'grill', 'fry', 'bake'],
        'reading_book': ['read', 'book', 'study', 'text'],
        'cleaning_shoes': ['clean', 'shoe', 'polish', 'brush'],
        'mowing_lawn': ['mow', 'lawn', 'grass', 'yardwork'],
        'carrying_baby': ['carry', 'baby', 'child', 'infant', 'hold'],
        'sanding_floor': ['sand', 'floor', 'refinish', 'woodwork'],
        'petting_animal': ['pet', 'animal', 'dog', 'cat', 'stroke'],
        'cleaning_floor': ['clean', 'floor', 'sweep', 'mop', 'vacuum'],
        'using_computer': ['comput', 'laptop', 'typ', 'screen', 'keyboard'],
        'folding_clothes': ['fold', 'clothe', 'laundry', 'shirt', 'towel'],
        'flipping_pancake': ['cook', 'pancake', 'flip', 'breakfast', 'kitchen', 'making'],
        'cleaning_windows': ['clean', 'window', 'wash', 'glass', 'pane'],
        'decorating_christmas_tree': ['decorat', 'christmas', 'tree', 'holiday', 'ornament'],
        'washing_dishes': ['wash', 'dish', 'plate', 'scrub', 'sink', 'kitchen', 'clean'],
    }

    # Add general terms if needed
    if 'cook' in query_action_lower or 'pancake' in query_action_lower:
         action_keywords.setdefault(query_action_key, []).extend(['cook', 'kitchen', 'meal', 'food', 'making'])
    if 'clean' in query_action_lower or 'wash' in query_action_lower:
         action_keywords.setdefault(query_action_key, []).extend(['clean', 'wash', 'tidy'])

    if query_action_key not in action_keywords:
        return None

    expected_keywords = action_keywords[query_action_key]
    match_found = any(keyword in activity_label_lower for keyword in expected_keywords)

    if query_action_lower.replace("_", "") in activity_label_lower.replace("_", ""):
         match_found = True

    return "Yes" if match_found else "No"


def evaluate_per_activity(data):
    """
    Calculates accuracy per activity label for multiple specified query types.
    Returns results suitable for writing to CSV.
    """
    activity_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    processed_queries = 0
    matched_target_queries = 0

    for entry in data:
        activity_label_orig = entry.get('activity', 'unknown_activity')
        activity_label_lower = activity_label_orig.lower()
        queries = entry.get('queries', [])

        if not isinstance(queries, list): continue

        for query_data in queries:
            processed_queries += 1
            if not isinstance(query_data, list) or len(query_data) < 2: continue

            query_str = str(query_data[0]).strip()
            response_str = str(query_data[1]).strip().lower()

            if query_str in TARGET_QUERIES:
                matched_target_queries += 1
                query_action_key = TARGET_QUERIES[query_str]

                ground_truth = determine_ground_truth(activity_label_orig, query_action_key)
                if ground_truth is None:
                    activity_stats[activity_label_orig][query_action_key]['skipped_indeterminate_gt'] += 1
                    continue

                prediction = "Unknown"
                if response_str.startswith("yes"): prediction = "Yes"
                elif response_str.startswith("no"): prediction = "No"

                stats = activity_stats[activity_label_orig][query_action_key]
                stats['total_evaluated'] += 1

                if prediction == "Unknown":
                    stats['unknown'] += 1
                elif prediction == ground_truth:
                    stats['correct'] += 1
                    if prediction == "Yes": stats['TP'] += 1
                    else: stats['TN'] += 1
                else:
                    stats['incorrect'] += 1
                    if prediction == "Yes": stats['FP'] += 1
                    else: stats['FN'] += 1

    print(f"\nProcessed {processed_queries} total query instances.")
    print(f"Matched {matched_target_queries} instances against target query list.")
    return activity_stats # Return the raw stats dictionary

def write_results_to_csv(evaluation_results, output_filename="activity_evaluation_results.csv"):
    """Writes the evaluation results dictionary to a CSV file."""
    fieldnames = [
        'activity_label',
        'query_action_key',
        'total_evaluated',
        'correct',
        'incorrect',
        'unknown',
        'tp',
        'tn',
        'fp',
        'fn',
        'accuracy' # Calculated field
    ]

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort for consistent output
        sorted_activities = sorted(evaluation_results.keys())
        for activity_label in sorted_activities:
            activity_data = evaluation_results[activity_label]
            sorted_query_actions = sorted(activity_data.keys())

            for query_action_key in sorted_query_actions:
                stats = activity_data[query_action_key]
                total_evaluated = stats.get('total_evaluated', 0)

                if total_evaluated > 0:
                    correct_preds = stats.get('correct', 0)
                    accuracy = correct_preds / total_evaluated if total_evaluated > 0 else 0

                    row = {
                        'activity_label': activity_label,
                        'query_action_key': query_action_key,
                        'total_evaluated': total_evaluated,
                        'correct': correct_preds,
                        'incorrect': stats.get('incorrect', 0),
                        'unknown': stats.get('unknown', 0),
                        'tp': stats.get('TP', 0),
                        'tn': stats.get('TN', 0),
                        'fp': stats.get('FP', 0),
                        'fn': stats.get('FN', 0),
                        'accuracy': f"{accuracy:.4f}" # Format accuracy for CSV
                    }
                    writer.writerow(row)
    print(f"\nResults successfully written to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy per activity label for specific query types and output to CSV.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("-o", "--output", default="activity_evaluation_results.csv", help="Path to the output CSV file.")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

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

    # Evaluate the data
    evaluation_results = evaluate_per_activity(video_data)

    # Write results to CSV
    write_results_to_csv(evaluation_results, args.output)

    # (Optional: Keep the console printing if you still want it)
    # print_results_to_console(evaluation_results) # You would need to create/adapt this function

if __name__ == "__main__":
    main()