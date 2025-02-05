import pandas as pd

# Load the CSV file with generated and updated keywords, including match data
df = pd.read_csv("/home/jovyan/trainingmodel/keywordscourse/flant5largert/test_keywords_with_matches.csv")

# Ensure the required columns exist
assert 'total_matches' in df.columns and 'total_updated_keywords' in df.columns, "Columns not found in the CSV file."

# Function to calculate match percentage
def match_percentage(row):
    total_matches = row['total_matches']
    total_generated = row['total_updated_keywords']
    
    # Avoid division by zero
    if total_generated == 0:
        return 0
    
    # Calculate percentage of matched keywords
    return (total_matches / total_generated) * 100

# Apply the match percentage calculation to the DataFrame
df['match_percentage'] = df.apply(match_percentage, axis=1)

# Define a function to count rows where match percentage exceeds a threshold
def count_matches_above_threshold(df, threshold):
    return len(df[df['match_percentage'] >= threshold])

# Print the counts for each threshold (10%, 20%, ... 100%)
for threshold in range(10, 110, 10):
    count = count_matches_above_threshold(df, threshold)
    print(f"Rows with match percentage (flant5large Updkw) >= {threshold}%: {count}")

# Save the updated DataFrame with the match percentage column back to the CSV file
df.to_csv("/home/jovyan/trainingmodel/keywordscourse/flant5largert/test_keywords_with_match_percentages_upd.csv", index=False)
