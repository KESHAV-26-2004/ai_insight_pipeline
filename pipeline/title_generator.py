# pipeline/title_generator.py
import os
import json
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ============================
# CONFIGURATION
# ============================
MODEL_PATH = "./title/model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# HELPER FUNCTIONS
# ============================
def summarize_csv_for_prompt(file_path):
    """
    Reads CSV and creates the specific summary format the model was trained on.
    """
    try:
        df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")
    except:
        df = pd.read_csv(file_path, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")

    cols = df.columns.tolist()
    summary = f"Columns: {', '.join(cols)}. Rows: {len(df)}. "
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) > 0:
        summary += f"Contains numeric fields such as {', '.join(numeric_cols)}."
    else:
        summary += "No numeric fields detected."

    return summary

def clean_generated_title(text):
    if "Title:" in text:
        text = text.split("Title:", 1)[1]
    return text.split("\n")[0].strip()

# ============================
# MAIN GENERATION FUNCTION
# ============================
def generate_and_save_title(csv_path, out_dir):
    """
    Loads the model, generates a title, and saves it to JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(csv_path).replace(".csv", "")
    out_file = os.path.join(out_dir, f"{base_name}_title.json")

    print(f"🧠 Loading Title Model from {MODEL_PATH} to {DEVICE}...")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model = model.to(DEVICE)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Ensure './title_model' folder exists in the root directory.")
        return

    # Generate Prompt
    print("🔍 Creating Summary for Prompt...")
    summary = summarize_csv_for_prompt(csv_path)
    prompt = f"Summary: {summary}\nTitle:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    print("⚡ Generating Title...")
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    final_title = clean_generated_title(decoded)

    # Save Result
    result_data = {
        "original_csv": csv_path,
        "dataset_name": base_name,
        "generated_title": final_title,
        "model_used": MODEL_PATH
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)

    print(f"✅ Title Generated: '{final_title}'")
    print(f"💾 Saved to: {out_file}")

    return final_title