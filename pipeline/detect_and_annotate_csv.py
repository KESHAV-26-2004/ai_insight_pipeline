import os, json, re, math, time
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import pipeline
from utils.io_utils import read_csv_safe, avg_word_count, unique_word_ratio, has_sentiment_keywords, clean_text_simple

# ---- Config paths ----
OPINION_MODEL_DIR = "./Sentiment/opinion_detector_model/final"
SENTIMENT_MODEL_DIR = "./Sentiment/sentiment_model/distilbert-base-uncased/final"
DEFAULT_OUT_DIR = "./results/enriched_csvs"
os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

# ============================================================
# 🔍 Detect existing sentiment-like columns (only text-based)
# ============================================================
def detect_existing_sentiment_cols(df, min_rows=30, required_frac=0.9):
    cand = []
    text_re = re.compile(r"^(positive|negative|neutral|pos|neg|neu)$", re.IGNORECASE)

    for col in df.columns:
        if col.endswith(("_sentiment_num", "_sentiment_confidence")):
            continue

        s = df[col].dropna()
        if s.shape[0] < min_rows:
            continue

        if s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.astype(str)
            matched = sample.str.strip().str.lower().apply(lambda x: bool(text_re.match(x)))
            frac = matched.mean() if len(matched) > 0 else 0.0
            if frac >= required_frac:
                cand.append((col, "text"))

    return cand


# ============================================================
# 🧩 Normalize an existing sentiment column
# ============================================================
def normalize_existing_sentiment(df, col):
    sent_col = f"{col}_sentiment"
    sent_num_col = f"{col}_sentiment_num"

    def map_text_label(x):
        if not isinstance(x, str):
            return None
        x0 = x.strip().lower()
        if x0 in ("positive", "pos", "+", "+1"):
            return "positive"
        if x0 in ("neutral", "neu", "n", "0"):
            return "neutral"
        if x0 in ("negative", "neg", "-", "-1"):
            return "negative"
        return None

    mapped = df[col].astype(str).apply(map_text_label)
    df[sent_col] = mapped.fillna("neutral")
    map_num = {"positive": 2, "neutral": 1, "negative": 0}
    df[sent_num_col] = df[sent_col].map(map_num).fillna(1).astype(int)
    return sent_col, sent_num_col


# ============================================================
# 🧠 Main function (optimized for speed)
# ============================================================
def detect_and_annotate_csv(csv_path, out_dir=DEFAULT_OUT_DIR,
                            opinion_model_dir=OPINION_MODEL_DIR,
                            sentiment_model_dir=SENTIMENT_MODEL_DIR,
                            sample_for_check=500,
                            opinion_threshold=0.25,
                            conf_threshold=0.5,
                            min_rows_sentiment_detect=30,
                            required_frac_existing=0.9,
                            validate_existing_with_model=False):

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(csv_path).replace(".csv", "")

    df = read_csv_safe(csv_path)

    # === Load dtype from profile (correct path resolution) ===

    # Step 1: folder containing the CSV
    csv_dir = os.path.dirname(csv_path)
    # → results/GI_cleaned_with_sentiment/cleaned

    # Step 2: dataset folder name
    dataset_dir = os.path.basename(os.path.dirname(csv_dir))
    # → Genshin Impact_cleaned_with_sentiment

    # Step 3: construct correct profile path
    profile_path = os.path.join(
        "results",
        dataset_dir,
        "profiles",
        f"{dataset_dir}_profile.json"
    )

    dtype_map = {}

    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            dtype_map = {col: info["dtype"] for col, info in profile_data["columns"].items()}
        print("📌 Loaded dtype_map from profile:", dtype_map)
    else:
        print("⚠️ No profile found at:", profile_path)


    print(f"📄 Loaded '{base}' — {df.shape[0]} rows, {df.shape[1]} cols")

    meta = {"file": base, "candidates": [], "analyzed": [], "skipped": [], "existing_sentiment": [], "derived_columns": []}

    # ============================================================
    # Step A — Detect existing sentiment-like columns
    # ============================================================
    existing = detect_existing_sentiment_cols(df, min_rows=min_rows_sentiment_detect, required_frac=required_frac_existing)
    if existing:
        print("🟢 Found existing sentiment-like columns:", existing)

        for col, _ in existing:
            sent_text_col = col
            sent_num_col = f"{col}_num"
            map_num = {"positive": 2, "neutral": 1, "negative": 0, "pos": 2, "neu": 1, "neg": 0}
            df[sent_num_col] = df[sent_text_col].astype(str).str.lower().map(map_num).fillna(1).astype(int)
            print(f"✅ Normalized '{sent_text_col}' → added numeric column '{sent_num_col}'")
            meta["existing_sentiment"].append({
                "column": sent_text_col,
                "numeric_version": sent_num_col,
                "kind": "text"
            })

        cleaned_dir = out_dir.replace("enriched", "cleaned")
        meta_path = os.path.join(cleaned_dir, f"{base}_targets_meta.json")
        os.makedirs(cleaned_dir, exist_ok=True)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                target_meta = json.load(f)
        else:
            target_meta = {
                "file": f"{base}.csv",
                "primary": None,
                "secondary": [],
                "type": "classification",
                "title": "Unknown",
                "auto_changes": [],
                "needs_review": False
            }

        for col, _ in existing:
            if col not in target_meta.get("secondary", []) and col != target_meta.get("primary"):
                target_meta["secondary"].append(col)
                print(f"➕ Added '{col}' to secondary targets in {base}_targets_meta.json")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(target_meta, f, indent=2)

        meta["note"] = "existing_sentiment_used"
        out_csv = os.path.join(out_dir, f"{base}_with_sentiment.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        out_json = os.path.join(out_dir, f"{base}_sentiment_meta.json")
        meta["runtime_sec"] = round(time.time() - t0, 2)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Existing sentiment column(s) normalized and saved → {out_csv}")
        print(f"📄 Updated meta → {meta_path}")
        return df, meta

    # ============================================================
    # Step B — Run models if no sentiment found
    # ============================================================
    candidates = []
    for col in df.columns:
        if dtype_map.get(col) == "datetime":
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        s = df[col].astype(str).dropna()
        if s.shape[0] == 0:
            continue
        aw = avg_word_count(df[col])
        ur = unique_word_ratio(df[col])
        kw = has_sentiment_keywords(df[col])
        if aw < 1.5 and ur > 0.9:
            continue
        if aw >= 2.0 or kw:
            candidates.append({"column": col, "avg_words": aw, "unique_word_ratio": ur, "keyword_signal": kw, "non_null_count": int(s.shape[0])})

    print(f"🧩 Text candidates found: {[c['column'] for c in candidates]}")
    meta["candidates"] = candidates

    # === 🔥 Optimization: Auto GPU + batch + caching ===
    device = 0 if torch.cuda.is_available() else -1
    batch_size = 64 if device == 0 else 16
    print(f"🔹 Loading models on {'GPU' if device==0 else 'CPU'} (batch_size={batch_size})...")
    opinion_clf = pipeline(
        "text-classification",
        model=opinion_model_dir,
        tokenizer=opinion_model_dir,
        device=device,
        return_all_scores=True,
        truncation=True,
        max_length=512
    )

    sentiment_clf = pipeline(
        "text-classification",
        model=sentiment_model_dir,
        tokenizer=sentiment_model_dir,
        device=device,
        return_all_scores=True,
        truncation=True,
        max_length=512
    )

    print("✅ Models loaded.")

    enriched_cols = []
    cache = {}

    for cand in candidates:
        col = cand["column"]
        print(f"\n🧠 Checking opinion in '{col}' ...")
        sample = df[col].dropna().astype(str).sample(min(sample_for_check, len(df[col].dropna())), random_state=1).tolist()
        preds = opinion_clf(sample)
        preds_flat = [max(p, key=lambda x: x["score"]) for p in preds]
        opinion_scores = [p["score"] for p in preds_flat if p["label"].lower() in ["1", "opinion", "label_1", "positive"]]
        opinion_frac = len(opinion_scores) / len(preds_flat)
        mean_conf = np.mean(opinion_scores) if opinion_scores else 0
        print(f"→ Opinion fraction={opinion_frac:.2f}, mean_conf={mean_conf:.2f}")

        if opinion_frac < opinion_threshold:
            print(f"⚪ '{col}' skipped (not opinion-rich enough).")
            meta["skipped"].append({col: {"opinion_frac": opinion_frac, "mean_conf": mean_conf}})
            continue

        # 🔥 Sentiment inference with cache and batch
        print(f"🟢 Running sentiment model on '{col}' ...")
        texts = df[col].fillna("").astype(str).apply(clean_text_simple).tolist()
        unique_texts = list(dict.fromkeys(texts))
        new_texts = [t for t in unique_texts if t not in cache]

        # process only uncached texts
        for i in tqdm(range(0, len(new_texts), batch_size), desc=f"Inferencing '{col}'"):
            batch = new_texts[i:i+batch_size]
            results = sentiment_clf(batch, truncation=True)
            for t, r in zip(batch, results):
                top = max(r, key=lambda x: x["score"])
                cache[t] = (top["label"].lower(), top["score"])

        # reconstruct labels from cache
        labels, confs = [], []
        label_text_map = {"label_0": "negative", "label_1": "neutral", "label_2": "positive",
                          "negative": "negative", "neutral": "neutral", "positive": "positive"}
        for txt in texts:
            raw_label, score = cache[txt]
            label = label_text_map.get(raw_label, "neutral")
            if score < conf_threshold:
                label = "neutral"
            labels.append(label)
            confs.append(score)

        df[f"{col}_sentiment"] = labels
        label_num_map = {"negative": 0, "neutral": 1, "positive": 2}
        df[f"{col}_sentiment_num"] = [label_num_map.get(l, 1) for l in labels]
        df[f"{col}_sentiment_confidence"] = confs

        enriched_cols.append(col)
        meta["analyzed"].append({"column": col, "opinion_frac": opinion_frac, "mean_conf": mean_conf,
                                 "rows_processed": len(df),
                                 "added_columns": [f"{col}_sentiment", f"{col}_sentiment_num", f"{col}_sentiment_confidence"]})

    # Save results (same)
    out_csv = os.path.join(out_dir, f"{base}_with_sentiment.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = os.path.join(out_dir, f"{base}_sentiment_meta.json")
    meta["runtime_sec"] = round(time.time() - t0, 2)
    meta["file"] = base
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Done! {len(enriched_cols)} columns labeled.")
    print(f"📁 Saved enriched CSV → {out_csv}")
    print(f"📄 Saved metadata → {out_json}")
    print(f"⚡ Runtime: {meta['runtime_sec']:.2f}s")

    return df, meta
