import json
import argparse
import os
from collections import defaultdict
import re

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

    Args:
        activity_label (str): The ground truth label for the activity.
        query_action_key (str): The specific action identifier (e.g., 'reading_book').

    Returns:
        str or None: 'Yes', 'No', or None if undetermined.
    """
    activity_label_lower = activity_label.lower()
    query_action_lower = query_action_key.lower()

    # --- Define Keyword relationships ---
    # This section requires careful thought based on your activity labels.
    # It maps the query_action_key to keywords expected in the activity_label.

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
        'using_computer': ['comput', 'laptop', 'typ', 'screen', 'keyboard'], # 'comput' matches computer/computing
        'folding_clothes': ['fold', 'clothe', 'laundry', 'shirt', 'towel'],
        'flipping_pancake': ['cook', 'pancake', 'flip', 'breakfast', 'kitchen', 'making'], # Related to cooking
        'cleaning_windows': ['clean', 'window', 'wash', 'glass', 'pane'],
        'decorating_christmas_tree': ['decorat', 'christmas', 'tree', 'holiday', 'ornament'],
        'washing_dishes': ['wash', 'dish', 'plate', 'scrub', 'sink', 'kitchen', 'clean'],
    }

    # Add general cooking/cleaning terms to relevant specific actions
    if 'cook' in query_action_lower or 'pancake' in query_action_lower:
         action_keywords.setdefault(query_action_key, []).extend(['cook', 'kitchen', 'meal', 'food', 'making'])
    if 'clean' in query_action_lower or 'wash' in query_action_lower:
         action_keywords.setdefault(query_action_key, []).extend(['clean', 'wash', 'tidy'])


    # --- Ground Truth Logic ---
    if query_action_key not in action_keywords:
        # If we haven't defined keywords for this action, we can't determine GT
        # This shouldn't happen if TARGET_QUERIES is the source, but safety check.
        return None

    expected_keywords = action_keywords[query_action_key]

    # Check if *any* of the relevant keywords for this action are in the activity label
    # More sophisticated logic could be used (e.g., requiring multiple keywords)
    match_found = any(keyword in activity_label_lower for keyword in expected_keywords)

    # Special handling: If activity label itself IS the action key (or very similar)
    # E.g., activity="mowing_lawn", query_action="mowing_lawn"
    if query_action_lower.replace("_", "") in activity_label_lower.replace("_", ""):
         match_found = True
    # Handle cases like activity="cooking" and query asks about "cooking_chicken"
    # We might default to "Yes" if the general category matches. Requires careful thought.
    if query_action_key == 'cooking_chicken' and 'cooking' in activity_label_lower and 'chicken' not in activity_label_lower:
         pass # Let the keyword check decide, or force Yes/No? Ambiguous.
         # For now, rely on keyword check above.

    return "Yes" if match_found else "No"


def evaluate_per_activity(data):
    """
    Calculates accuracy per activity label for multiple specified query types.
    """
    # Structure: activity_stats[activity_label][query_action_key][metric_type] = count
    activity_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    processed_queries = 0
    matched_target_queries = 0

    for entry in data:
        activity_label_orig = entry.get('activity', 'unknown_activity') # Keep original case for display
        activity_label_lower = activity_label_orig.lower()
        queries = entry.get('queries', [])

        if not isinstance(queries, list):
            continue

        for query_data in queries:
            processed_queries += 1
            if not isinstance(query_data, list) or len(query_data) < 2:
                continue

            query_str = str(query_data[0]).strip() # Match exact target queries
            response_str = str(query_data[1]).strip().lower()

            # --- Check if this query is one we want to evaluate ---
            if query_str in TARGET_QUERIES:
                matched_target_queries += 1
                query_action_key = TARGET_QUERIES[query_str] # Get our internal action key

                # --- Determine Ground Truth ---
                ground_truth = determine_ground_truth(activity_label_orig, query_action_key)

                if ground_truth is None:
                    activity_stats[activity_label_orig][query_action_key]['skipped_indeterminate_gt'] += 1
                    continue

                # --- Determine Prediction ---
                prediction = None
                if response_str.startswith("yes"):
                    prediction = "Yes"
                elif response_str.startswith("no"):
                    prediction = "No"
                else:
                    prediction = "Unknown"

                # --- Increment Counts ---
                stats = activity_stats[activity_label_orig][query_action_key] # Shortcut
                stats['total_evaluated'] += 1

                if prediction == "Unknown":
                    stats['unknown'] += 1
                elif prediction == ground_truth:
                    stats['correct'] += 1
                    if prediction == "Yes": stats['TP'] += 1
                    else: stats['TN'] += 1
                else: # Incorrect prediction
                    stats['incorrect'] += 1
                    if prediction == "Yes": stats['FP'] += 1
                    else: stats['FN'] += 1
            # Else: Query is not in TARGET_QUERIES, so ignore it for this evaluation

    print(f"\nProcessed {processed_queries} total query instances.")
    print(f"Matched {matched_target_queries} instances against target query list.")
    return activity_stats


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy per activity label for specific query types.")
    parser.add_argument("input_file", help="Path to the input JSON file.")

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

    # --- Print Results ---
    print("\n--- Evaluation Metrics Per Activity Label ---")
    if not evaluation_results:
        print("No activities found or no relevant target queries processed.")
        return

    # Sort activities alphabetically for consistent output
    sorted_activities = sorted(evaluation_results.keys())

    for activity_label in sorted_activities:
        print(f"\nActivity: '{activity_label}'")
        activity_data = evaluation_results[activity_label]
        printed_activity_header = False

        # Sort query actions alphabetically within each activity
        sorted_query_actions = sorted(activity_data.keys())

        for query_action_key in sorted_query_actions:
            stats = activity_data[query_action_key]
            total_evaluated = stats.get('total_evaluated', 0)

            if total_evaluated > 0:
                 # Only print activity header once if there's data
                #if not printed_activity_header:
                #    print(f"\nActivity: '{activity_label}'")
                #    printed_activity_header = True

                # Calculate accuracy (penalizing unknowns)
                correct_preds = stats.get('correct', 0)
                accuracy = correct_preds / total_evaluated if total_evaluated > 0 else 0

                print(f"  Query Type: '{query_action_key}'")
                print(f"    Total Evaluated: {total_evaluated}")
                # print(f"    Correct:         {correct_preds}")
                # print(f"    Incorrect:       {stats.get('incorrect', 0)}")
                print(f"    Unknown Responses: {stats.get('unknown', 0)}")
                print(f"    Accuracy:        {accuracy:.4f}")
                print(f"    TP (Correct 'Yes'):   {stats.get('TP', 0)}")
                print(f"    TN (Correct 'No'):    {stats.get('TN', 0)}")
                print(f"    FP (Incorrect 'Yes'): {stats.get('FP', 0)}")
                print(f"    FN (Incorrect 'No'):  {stats.get('FN', 0)}")
                print(f"    -------------------------------")

        #if not printed_activity_header:
        #     print(f"\nActivity: '{activity_label}' (No target queries evaluated)")


if __name__ == "__main__":
    main()