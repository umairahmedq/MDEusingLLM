import pandas as pd

# Load the CSV file with matches and counts
df = pd.read_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/test_keywords_with_matches.csv")

# Ensure required columns exist
assert 'total_matches' in df.columns, "Column 'total_matches' not found in the file."
assert 'total_generated_keywords' in df.columns, "Column 'total_generated_keywords' not found in the file."
assert 'total_updated_keywords' in df.columns, "Column 'total_updated_keywords' not found in the file."

# Calculate precision, recall, and F1 score for each row
def calculate_metrics(row):
    matches = row['total_matches']
    generated = row['total_generated_keywords']
    original = row['total_updated_keywords']

    # Avoid division by zero
    precision = matches / generated if generated > 0 else 0
    recall = matches / original if original > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return pd.Series([precision, recall, f1_score])

# Apply the function to calculate metrics
df[['precision', 'recall', 'f1_score']] = df.apply(calculate_metrics, axis=1)

# Calculate overall metrics
total_matches = df['total_matches'].sum()
total_generated_keywords = df['total_generated_keywords'].sum()
total_updated_keywords = df['total_updated_keywords'].sum()

# Overall precision, recall, and F1 score
overall_precision = total_matches / total_generated_keywords if total_generated_keywords > 0 else 0
overall_recall = total_matches / total_updated_keywords if total_updated_keywords > 0 else 0
overall_f1_score = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

# Print overall metrics
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall F1 Score: {overall_f1_score:.4f}")

# Save the DataFrame with the new columns to a new CSV file
output_file = "/home/jovyan/trainingmodel/keywordscourse/flant5small/test_keywords_with_scores.csv"
df.to_csv(output_file, index=False)

# Save overall metrics to a summary file
summary_file = "/home/jovyan/trainingmodel/keywordscourse/flant5small/eval_summary_scores.txt"
with open(summary_file, "w") as f:
    f.write(f"Overall Precision: {overall_precision:.4f}\n")
    f.write(f"Overall Recall: {overall_recall:.4f}\n")
    f.write(f"Overall F1 Score: {overall_f1_score:.4f}\n")

print(f"Scores calculated and saved to '{output_file}'.")
print(f"Overall scores saved to '{summary_file}'.")
