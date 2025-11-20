# utils/io_utils.py

import os
import json
import pandas as pd
import numpy as np
import re

# -----------------------------------
# 🧱 General Utilities
# -----------------------------------

def ensure_dir(path):
    """Ensure a directory exists. path can be a file path or a folder path."""
    folder = path if os.path.isdir(path) or path.endswith(os.sep) else os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

def read_csv_safe(path):
    """Safely read CSV with multiple encoding attempts."""
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    raise ValueError(f"❌ Unable to read CSV: {path}")

# -----------------------------------
# 🧹 Text Cleaning + Simple NLP Metrics
# -----------------------------------

def clean_text_simple(s):
    """Basic cleaner for text fields."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\S+", "", s)
    s = re.sub(r"[^a-zA-Z0-9\s\.\,\?\!\-\'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def avg_word_count(series, sample_size=500):
    s = series.dropna().astype(str)
    if s.shape[0] == 0:
        return 0.0
    sample = s.sample(min(len(s), sample_size), random_state=1)
    return np.mean([len(x.split()) for x in sample])

def unique_word_ratio(series, sample_size=500):
    s = series.dropna().astype(str)
    if s.shape[0] == 0:
        return 0.0
    sample = s.sample(min(len(s), sample_size), random_state=1)
    words = " ".join(sample.tolist()).split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

SENTIMENT_KEYWORDS = {
    "good","great","bad","terrible","excellent","worst","love","hate",
    "amazing","awful","disappointed","satisfied","recommend","poor","fantastic","nice"
}

def has_sentiment_keywords(series, sample_size=500):
    s = series.dropna().astype(str)
    if s.shape[0] == 0:
        return False
    sample = s.sample(min(len(s), sample_size), random_state=1)
    cnt = 0
    for text in sample:
        tokens = set(t.lower().strip(".,!?\'\"") for t in text.split())
        if len(SENTIMENT_KEYWORDS.intersection(tokens)) > 0:
            cnt += 1
    return (cnt / len(sample)) > 0.01  # >1% rows contain keyword

# -----------------------------------
# ⚙️ Feature Config Generator
# -----------------------------------

def make_feature_config(dataset_name,
                        results_root="./results",
                        labelled_json="./data/labelled_datasets.json"):
    """
    Automatically builds the config JSON used by feature_engineer.main().
    Will write it to ./configs/<dataset>_feature_config.json and return its path.
    """
    os.makedirs("./configs", exist_ok=True)

    base = dataset_name.replace(".csv", "")
    dataset_dir = os.path.join(results_root, base)
    enriched_dir = os.path.join(dataset_dir, "enriched")
    profiles_dir = os.path.join(dataset_dir, "profiles")
    cleaned_dir = os.path.join(dataset_dir, "cleaned")
    meta_dir = os.path.join(dataset_dir, "meta")

    os.makedirs(enriched_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    config = {
        "input_csv": os.path.join(enriched_dir, f"{base}_cleaned_with_sentiment.csv"),
        "profile_json": os.path.join(profiles_dir, f"{base}_profile.json"),
        "labelled_json": os.path.join(cleaned_dir, f"{base}_cleaned_targets_meta.json"),
        "output_csv": os.path.join(dataset_dir, "features", f"{base}_features.csv"),
        "output_meta": os.path.join(meta_dir, f"{base}_encoders.json")
    }

    os.makedirs(os.path.dirname(config["output_csv"]), exist_ok=True)

    config_path = f"./configs/{base}_feature_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Feature config saved → {config_path}")
    return config_path
