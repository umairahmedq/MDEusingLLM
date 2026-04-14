import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Load fine-tuned model
# =========================
model_path = "/home/jovyan/shared/Umair/models/finetuned/dskeywords/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Fix padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

# =========================
# Load test CSV
# =========================
df = pd.read_csv("/home/jovyan/trainingmodel/keywordscourse/dsft/test_keywords.csv")

assert 'updated_description' in df.columns

# =========================
# Post-processing cleaner
# =========================
def clean_keywords(text):
    # Remove unwanted trailing sections
    text = re.split(r"###|Explanation|Additional|Field Name", text)[0]

    # Split into keywords
    keywords = [k.strip() for k in text.split(",")]

    clean = []
    seen = set()

    for kw in keywords:
        if not kw:
            continue

        # Remove long phrases (>3 words)
        if len(kw.split()) > 3:
            continue

        # Normalize for duplicate removal
        kw_lower = kw.lower()

        if kw_lower not in seen:
            seen.add(kw_lower)
            clean.append(kw)

    # Limit to max 12 keywords
    return ", ".join(clean[:12])

# =========================
# Generation function
# =========================
def generate_keywords(description):
    # Keep structure similar to fine-tuning
    prompt = (
        "### Instruction:\n"
        "Generate 3 to 12 keywords.\n"
        "Return only keywords separated by commas. No explanation.\n\n"
        f"### Input:\n{description}\n\n"
        "### Output:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.5,           # more focused
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.3,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract generated part
    if "### Output:" in decoded:
        generated = decoded.split("### Output:")[-1].strip()
    else:
        generated = decoded.strip()

    # Clean output
    generated = clean_keywords(generated)

    return generated

# =========================
# Run generation
# =========================
generated_list = []

for index, row in df.iterrows():
    desc = row['updated_description']

    keywords = generate_keywords(desc)
    print(f"\nGenerated: {keywords}")

    generated_list.append(keywords)

# Add to dataframe
df["generated_keywords"] = generated_list

# =========================
# Save results
# =========================
output_path = "/home/jovyan/trainingmodel/keywordscourse/dsft/test_keywords_results.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Saved results to: {output_path}")