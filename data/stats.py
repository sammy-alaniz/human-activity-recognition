import json
import argparse
import os
from collections import defaultdict

def calculate_metrics(tp, tn, fp, fn, unknown):
    """Calculates accuracy (modified), precision, recall, and F1-score."""
    # Total predictions where Yes/No was determined
    total_determined = tp + tn + fp + fn
    # Total queries evaluated including unknowns
    total_evaluated = total_determined + unknown

    # Modified accuracy penalizes unknowns
    modified_accuracy = (tp + tn) / total_evaluated if total_evaluated > 0 else 0

    # Standard Precision, Recall, F1 are based on determined predictions
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "modified_accuracy": modified_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unknown_responses": unknown,
        "total_evaluated": total_evaluated
    }

def evaluate_labels(data):
    """
    Calculates TP, TN, FP, FN, and Unknowns for cooking and cleaning queries.

    Args:
        data (list): List of dictionaries loaded from the JSON file.

    Returns:
        dict: A dictionary containing the metrics for each query type.
    """
    counts = defaultdict(lambda: defaultdict(int)) # e.g., counts['cooking']['TP'] = 0

    for entry in data:
        activity = entry.get('activity', '').lower()
        queries = entry.get('queries', [])

        if not isinstance(queries, list):
            continue

        for query_data in queries:
            if not isinstance(query_data, list) or len(query_data) < 2:
                continue

            query_str = str(query_data[0]).lower()
            response_str = str(query_data[1]).strip().lower()

            # --- Determine Query Type and Ground Truth ---
            query_type = None
            ground_truth = None

            if "cooking" in query_str:
                query_type = "cooking"
                ground_truth = "Yes" if "cooking" in activity else "No"
            elif "cleaning" in query_str:
                query_type = "cleaning"
                is_cleaning_activity = "cleaning" in activity or "washing" in activity
                ground_truth = "Yes" if is_cleaning_activity else "No"

            # Only proceed if it's a query type we are evaluating
            if not query_type or not ground_truth:
                 continue

            # --- Determine Prediction (Extracted Label) ---
            prediction = None
            if response_str.startswith("yes"):
                prediction = "Yes"
            elif response_str.startswith("no"):
                prediction = "No"
            else:
                # Count as Unknown if neither Yes nor No is found
                prediction = "Unknown"


            # --- Increment Counts ---
            if prediction == "Yes":
                if ground_truth == "Yes":
                    counts[query_type]['TP'] += 1
                else: # ground_truth == "No"
                    counts[query_type]['FP'] += 1
            elif prediction == "No":
                if ground_truth == "No":
                    counts[query_type]['TN'] += 1
                else: # ground_truth == "Yes"
                    counts[query_type]['FN'] += 1
            elif prediction == "Unknown":
                 counts[query_type]['Unknown'] += 1


    # --- Calculate Metrics for each type ---
    results = {}
    for q_type in counts:
        tp = counts[q_type]['TP']
        tn = counts[q_type]['TN']
        fp = counts[q_type]['FP']
        fn = counts[q_type]['FN']
        unknown = counts[q_type]['Unknown'] # Get the count of unknowns
        results[q_type] = calculate_metrics(tp, tn, fp, fn, unknown)

    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate Modified Accuracy, Precision, and Recall from JSON video query data.")
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
    evaluation_results = evaluate_labels(video_data)

    # Print results
    print("\n--- Evaluation Metrics ---")
    if not evaluation_results:
        print("No cooking or cleaning queries found to evaluate.")
        return

    for query_type, metrics in evaluation_results.items():
        print(f"\nMetrics for '{query_type.capitalize()}' Queries:")
        print(f"  True Positives (TP):    {metrics['tp']}")
        print(f"  True Negatives (TN):    {metrics['tn']}")
        print(f"  False Positives (FP):   {metrics['fp']}")
        print(f"  False Negatives (FN):   {metrics['fn']}")
        print(f"  Unknown Responses:      {metrics['unknown_responses']}") # Report unknowns
        print(f"  -------------------------------")
        # Label the accuracy clearly
        print(f"  Accuracy (penalizing unknowns): {metrics['modified_accuracy']:.4f}")
        # Standard Precision/Recall/F1
        print(f"  Precision:                    {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity):         {metrics['recall']:.4f}")
        print(f"  F1-Score:                     {metrics['f1_score']:.4f}")
        print(f"  Total Evaluated:              {metrics['total_evaluated']}")


if __name__ == "__main__":
    main()