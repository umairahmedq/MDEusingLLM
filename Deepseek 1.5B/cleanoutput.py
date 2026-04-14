import pandas as pd
import re

# =========================
# Load file
# =========================
input_path = "/home/jovyan/trainingmodel/keywordscourse/dsft/test_keywords_results.csv"
df = pd.read_csv(input_path)

# =========================
# Clean function (robust)
# =========================
def clean_keywords(text):
    if pd.isna(text):
        return text

    text = str(text).strip()

    # Remove leading numbers like:
    # "1 arret, bus..." OR "1    arret, bus..."
    text = re.sub(r'^\s*\d+\s+', '', text)

    return text.strip()

# =========================
# Apply ONLY to generated_keywords
# =========================
df["generated_keywords"] = df["generated_keywords"].apply(clean_keywords)

# =========================
# Save cleaned file
# =========================
output_path = "/home/jovyan/trainingmodel/keywordscourse/dsft/test_keywords_results_cleaned.csv"
df.to_csv(output_path, index=False)

print("✅ Cleaned keywords saved:", output_path)