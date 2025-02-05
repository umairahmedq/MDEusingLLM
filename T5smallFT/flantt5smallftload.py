import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model_path = '/home/jovyan/shared/Umair/models/finetuned/flant5smallkeywordsc'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Load the CSV file
df = pd.read_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/test_keywords.csv")  # Replace with your actual CSV file path

# Ensure the columns you need exist
assert 'updated_description' in df.columns and 'updated_keywords' in df.columns, "Columns not found in the CSV file."

# Function to generate keywords from descriptions
def generate_keywords(description):
    # Prepare the input with the prompt
    input_text = "Generate 3 to 12 keywords for the following dataset description. Description: " + description
    
    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    # Generate keywords using the model
    output_ids = model.generate(input_ids, max_new_tokens=50, repetition_penalty=3.0, do_sample = True, temperature=0.7, no_repeat_ngram_size=3)
    
    # Decode the generated tokens to get the keywords
    generated_keywords = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_keywords

# Iterate over the DataFrame and generate keywords for each description
for index, row in df.iterrows():
    original_description = row['updated_description']
    
    # Generate keywords for the description
    generated_keywords = generate_keywords(original_description)
    print(generated_keywords)
    # Store the generated keywords in the DataFrame in a new column
    df.at[index, 'generated_keywords'] = generated_keywords

# Save the DataFrame with the generated keywords back to the same CSV file
df.to_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/test_keywords_results.csv", index=False)

print("Generated keywords have been saved in the same file under the 'generated_keywords' column.")
