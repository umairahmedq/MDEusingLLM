import pandas as pd
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import torch

# =========================
# Load pre-split CSVs
# =========================
train_path = "/home/jovyan/trainingmodel/keywordscourse/dsft/train_keywords.csv"
eval_path  = "/home/jovyan/trainingmodel/keywordscourse/dsft/eval_keywords.csv"
test_path  = "/home/jovyan/trainingmodel/keywordscourse/dsft/test_keywords.csv"

train_df = pd.read_csv(train_path)
eval_df  = pd.read_csv(eval_path)
test_df  = pd.read_csv(test_path)

# Validate columns
for df in [train_df, eval_df, test_df]:
    assert 'updated_description' in df.columns
    assert 'updated_keywords' in df.columns

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset  = Dataset.from_pandas(eval_df)
test_dataset  = Dataset.from_pandas(test_df)

# =========================
# Load model + tokenizer
# =========================
# BEST: use HF repo (avoids broken local paths)
model_path = "/home/jovyan/shared/Umair/models/DeepSeek-R1-Distill-Qwen-1.5B/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Fix padding for causal LM
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =========================
# Preprocessing (FIXED)
# =========================
def preprocess_function(examples):
    texts = []

    for desc, keywords in zip(examples['updated_description'], examples['updated_keywords']):
        text = (
            "### Instruction:\n"
            "Generate 3 to 12 keywords.\n\n"
            f"### Input:\n{desc}\n\n"
            f"### Output:\n{keywords}"
        )
        texts.append(text)

    tokenized = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Labels = input_ids (causal LM training)
    labels = []
    for seq in tokenized["input_ids"]:
        labels.append([
            token if token != tokenizer.pad_token_id else -100
            for token in seq
        ])

    tokenized["labels"] = labels
    return tokenized

# Apply preprocessing
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset  = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

# =========================
# Training arguments
# =========================
training_args = TrainingArguments(
    output_dir='/home/jovyan/shared/Umair/models/finetuned/dskeywords/results',
    learning_rate=4e-5,
    per_device_train_batch_size=4,   # safer for GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=7,              # start smaller, increase later
    weight_decay=0.02,
    logging_dir='/home/jovyan/shared/Umair/models/finetuned/dskeywords/logs',
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",           # FIXED (no warning)
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),  # safer check
    save_total_limit=2
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save
# =========================
model.save_pretrained('/home/jovyan/shared/Umair/models/finetuned/dskeywords')
tokenizer.save_pretrained('/home/jovyan/shared/Umair/models/finetuned/dskeywords')