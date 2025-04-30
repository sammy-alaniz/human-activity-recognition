import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("activity_evaluation_results.csv") # Or your specific filename

# --- 1. Overall Metrics ---

# Calculate overall TP, TN, FP, FN by summing across all rows
total_tp = df['tp'].sum()
total_tn = df['tn'].sum()
total_fp = df['fp'].sum()
total_fn = df['fn'].sum()
total_unknown = df['unknown'].sum()
total_determined = total_tp + total_tn + total_fp + total_fn
total_evaluated_overall = total_determined + total_unknown

# Overall Accuracy (penalizing unknowns)
overall_accuracy = (total_tp + total_tn) / total_evaluated_overall if total_evaluated_overall > 0 else 0
print(f"Overall Accuracy (penalizing unknowns): {overall_accuracy:.4f}")

# Overall Precision (Micro-averaged)
overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
print(f"Overall Precision: {overall_precision:.4f}")

# Overall Recall (Micro-averaged)
overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
print(f"Overall Recall: {overall_recall:.4f}")

# Overall F1-Score (Micro-averaged)
overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
print(f"Overall F1-Score: {overall_f1:.4f}")

print(f"Total Unknown Responses: {total_unknown}")


# --- 2. Performance Per Activity ---

# Group by activity and calculate average accuracy or sum TP/TN/FP/FN
activity_perf = df.groupby('activity_label').agg(
    avg_accuracy=('accuracy', 'mean'),
    total_tp=('tp', 'sum'),
    total_tn=('tn', 'sum'),
    total_fp=('fp', 'sum'),
    total_fn=('fn', 'sum'),
    total_eval=('total_evaluated', 'sum')
).sort_values(by='avg_accuracy', ascending=False)

print("\n--- Performance per Activity (Avg. Accuracy across Query Types) ---")
print(activity_perf)

# Plot Accuracy per Activity
plt.figure(figsize=(12, 8))
sns.barplot(x=activity_perf.index, y=activity_perf['avg_accuracy'])
plt.title('Average Accuracy per Activity Label')
plt.xlabel('Activity Label')
plt.ylabel('Average Accuracy')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('accuracy_per_activity.png') # Save the plot
# plt.show()


# --- 3. Performance Per Query Type ---

query_perf = df.groupby('query_action_key').agg(
    avg_accuracy=('accuracy', 'mean'),
    total_tp=('tp', 'sum'),
    total_tn=('tn', 'sum'),
    total_fp=('fp', 'sum'),
    total_fn=('fn', 'sum'),
    total_eval=('total_evaluated', 'sum')
).sort_values(by='avg_accuracy', ascending=False)

print("\n--- Performance per Query Type (Avg. Accuracy across Activities) ---")
print(query_perf)

# Plot Accuracy per Query Type
plt.figure(figsize=(12, 6))
sns.barplot(x=query_perf.index, y=query_perf['avg_accuracy'])
plt.title('Average Accuracy per Query Type')
plt.xlabel('Query Type (Action Key)')
plt.ylabel('Average Accuracy')
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig('accuracy_per_query_type.png') # Save the plot
# plt.show()

# --- 4. Confusion Matrix / Heatmap Ideas ---
# A traditional confusion matrix (Predicted Activity vs True Activity) isn't directly possible
# as the script doesn't predict the overall activity.
# Alternative: Heatmap of accuracy for Activity vs Query Type

try:
    # Pivot table for heatmap
    heatmap_data = df.pivot_table(index='activity_label', columns='query_action_key', values='accuracy')

    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title('Accuracy Heatmap: Activity Label vs. Query Type')
    plt.xlabel('Query Type (Action Key)')
    plt.ylabel('True Activity Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('accuracy_heatmap_activity_vs_query.png')
    # plt.show()
except Exception as e:
    print(f"\nCould not generate heatmap, possibly due to data shape or missing values: {e}")


# --- Further Analysis Ideas ---
# - Analyze activities/queries with the lowest accuracy.
# - Look at high FP rates (hallucinations) per activity.
# - Look at high FN rates (missed detections) per activity/query.
# - If you refined the ground truth logic, rerun the evaluation script and compare.