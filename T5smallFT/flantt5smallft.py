import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load the CSV file
df = pd.read_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/keywords.csv")  # Replace with your actual CSV file path

# Ensure the columns you need exist
assert 'updated_description' in df.columns and 'updated_keywords' in df.columns, "Columns not found in the CSV file."

# Load the tokenizer and model
model_path = "/home/jovyan/shared/Umair/models/flant5small/models--google--flan-t5-small/snapshots/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Filter rows where tokenized updated_descriptions have no more than 400 tokens
def filter_by_token_length(text, max_length=400):
    tokenized = tokenizer(text, truncation=True, return_tensors="pt")
    return len(tokenized['input_ids'][0]) <= max_length

df_filtered = df[df['updated_description'].apply(lambda x: filter_by_token_length(x))]

# Create a Hugging Face Dataset from the filtered DataFrame
dataset = Dataset.from_pandas(df_filtered[['updated_description', 'updated_keywords']])

# Split dataset into 80% training and 20% temp
temp_split = dataset.train_test_split(test_size=0.2)

# Further split temp into 50% evaluation and 50% test (10% of original dataset each)
eval_test_split = temp_split['test'].train_test_split(test_size=0.5)
train_dataset = temp_split['train']
eval_dataset = eval_test_split['train']
test_dataset = eval_test_split['test']

# Convert the Hugging Face datasets back to Pandas DataFrames to save as CSV
train_df = pd.DataFrame(train_dataset)
eval_df = pd.DataFrame(eval_dataset)
test_df = pd.DataFrame(test_dataset)

# Save to CSV files
train_df.to_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/train_keywords.csv", index=False)
eval_df.to_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/eval_keywords.csv", index=False)
test_df.to_csv("/home/jovyan/trainingmodel/keywordscourse/flant5small/test_keywords.csv", index=False)

# Now proceed with tokenizing and training
def preprocess_function(examples):
    # Tokenize the inputs and targets
    inputs = ["Generate 3 to 12 keywords for the following dataset description. Description: " + desc for desc in examples['updated_description']]
    model_inputs = tokenizer(inputs, max_length=462, truncation=True, padding="max_length")

    # Tokenize targets (updated_keywords)
    labels = tokenizer(examples['updated_keywords'], max_length=50, truncation=True, padding="max_length", text_target=examples['updated_keywords'])

    # Add labels to the model inputs
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# Apply preprocessing to both training and evaluation datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/home/jovyan/shared/Umair/models/finetuned/flant5smallkeywordsc/results',
    learning_rate=4e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=17,
    weight_decay=0.02,
    logging_dir='/home/jovyan/shared/Umair/models/finetuned/flant5smallkeywordsc/logs',
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    load_best_model_at_end=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the final model
model.save_pretrained('/home/jovyan/shared/Umair/models/finetuned/flant5smallkeywordsc')
tokenizer.save_pretrained('/home/jovyan/shared/Umair/models/finetuned/flant5smallkeywordsc')
