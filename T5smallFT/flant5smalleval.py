from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the CSV file with generated and updated keywords
df = pd.read_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/test_keywords_results.csv")

# Ensure the required columns exist
assert 'generated_keywords' in df.columns and 'updated_keywords' in df.columns, "Columns not found in the CSV file."

# Initialize ROUGE scorer
rouge = Rouge()

# Function to calculate ROUGE-1 F1 score for individual keywords
def rouge_scores_keyword(original_keyword, generated_keyword):
    scores = rouge.get_scores(generated_keyword, original_keyword, avg=True)
    rouge_1 = scores['rouge-1']['f']  # ROUGE-1 F1-score for unigram match
    return rouge_1

# Function to calculate Cosine Similarity for individual keywords
def cosine_similarity_keyword(original_keyword, generated_keyword):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([original_keyword, generated_keyword])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
    return cosine_sim[0][0]

# Function to split keywords, compare them, and count matches
def evaluate_row_keywords(original_keywords, generated_keywords, threshold=0.5):
    # Split the keywords into lists
    original_list = original_keywords.split(", ")
    generated_list = generated_keywords.split(", ")
    
    match_count = 0  # Count of matched keywords
    
    # Compare each generated keyword with each original keyword
    for gen_keyword in generated_list:
        matched = False
        for orig_keyword in original_list:
            # Calculate ROUGE-1 score
            rouge_1 = rouge_scores_keyword(orig_keyword, gen_keyword)
            
            # Calculate Cosine Similarity
            cosine_sim = cosine_similarity_keyword(orig_keyword, gen_keyword)
            
            # Check if any score is above the threshold
            if rouge_1 >= threshold or cosine_sim >= threshold:
                matched = True
                break  # Stop once a match is found for the current generated keyword
        
        if matched:
            match_count += 1
    
    # Return the total match count, total generated keywords, and total original keywords
    return match_count, len(generated_list), len(original_list)

# Create new columns for storing the counts
df['total_matches'] = 0
df['total_generated_keywords'] = 0
df['total_updated_keywords'] = 0

# Iterate over each row and compare keywords
for index, row in df.iterrows():
    original_keywords = row['updated_keywords']
    generated_keywords = row['generated_keywords']
    
    # Evaluate the row's keywords
    matches, total_generated, total_original = evaluate_row_keywords(original_keywords, generated_keywords)
    
    # Store the counts in the DataFrame
    df.at[index, 'total_matches'] = matches
    df.at[index, 'total_generated_keywords'] = total_generated
    df.at[index, 'total_updated_keywords'] = total_original

# Save the DataFrame with the new columns back to the CSV file
df.to_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/test_keywords_with_matches.csv", index=False)

print("Evaluation complete. Matches and counts have been saved in 'eval_keywords_with_matches.csv'.")
