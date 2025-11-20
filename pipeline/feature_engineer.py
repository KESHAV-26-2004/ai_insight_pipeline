# ==============================
# feature_engineer.py (Improved Pipeline Version)
# ==============================

import os
import re
import json
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.io_utils import read_csv_safe


def main(config_path):
    """
    Load configuration from JSON and execute feature engineering.
    Config fields:
      input_csv, profile_json, labelled_json, output_csv, output_meta
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    input_csv = cfg["input_csv"]
    profile_path = cfg["profile_json"]
    labelled_path = cfg["labelled_json"]
    out_csv = cfg["output_csv"]
    out_meta = cfg["output_meta"]

    print(f"🚀 Running feature engineering on: {input_csv}")

    # read input CSV
    df = read_csv_safe(input_csv)

    # preserve original columns list (before we add engineered columns)
    original_columns_on_load = list(df.columns)

    # load profile (optional)
    profile = json.load(open(profile_path, "r", encoding="utf-8")) if os.path.exists(profile_path) else None

    # labelled can be either dict (single entry) or list (legacy)
    labelled = None
    if os.path.exists(labelled_path):
        with open(labelled_path, "r", encoding="utf-8") as f:
            labelled = json.load(f)
    base = os.path.basename(input_csv).replace(".csv", "")

    # Handle new format: single dict
    if isinstance(labelled, dict):
        entry = labelled
    # Handle old format: list of dataset entries
    elif isinstance(labelled, list):
        entry = next((it for it in labelled if it.get("file") in [f"{base}.csv", base]), {})
    # Fallback
    else:
        entry = {}

    # Primary & secondary targets (normalize secondary into a list)
    primary = entry.get("primary")
    secondary = entry.get("secondary")
    if isinstance(secondary, str):
        secondary = [secondary]
    elif secondary is None:
        secondary = []
    elif not isinstance(secondary, list):
        # try to coerce other types
        try:
            secondary = list(secondary)
        except Exception:
            secondary = []

    print(f"Primary: {primary}, Secondary: {secondary}")

    # -------------------------------------
    # 1️⃣ Column Classification (using profile info if available)
    # -------------------------------------
    def classify_columns(df, profile=None):
        numeric_cols, categorical_cols, text_cols, datetime_cols = [], [], [], []

        for col in df.columns:
            if profile and col in profile.get("columns", {}):
                dtype = profile["columns"][col]["dtype"]
                if dtype == "numeric":
                    numeric_cols.append(col)
                    continue
                elif dtype == "datetime":
                    datetime_cols.append(col)
                    continue
                elif dtype == "text":
                    text_cols.append(col)
                    continue
                elif dtype == "categorical":
                    categorical_cols.append(col)
                    continue

            # fallback detection
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(s):
                datetime_cols.append(col)
            elif s.dtype == object or pd.api.types.is_string_dtype(s):
                avg_words = np.mean([
                    len(str(x).split()) for x in s.dropna().sample(min(200, len(s)), random_state=1)
                ]) if len(s.dropna()) > 0 else 0
                unique_ratio = s.nunique() / max(1, len(s))
                if avg_words < 3 and s.nunique() <= 30:
                    categorical_cols.append(col)
                else:
                    text_cols.append(col)
        return numeric_cols, categorical_cols, text_cols, datetime_cols

    numeric_cols, categorical_cols, text_cols, datetime_cols = classify_columns(df, profile)
    print(f"🔢 Numeric: {numeric_cols}")
    print(f"🔣 Categorical: {categorical_cols}")
    print(f"💬 Text: {text_cols}")
    print(f"🕒 Datetime: {datetime_cols}")

    # -------------------------------------
    # Remove primary/secondary from transformation lists (we must preserve them)
    # -------------------------------------
    def remove_targets_from_lists(primary, secondary, numeric_cols, categorical_cols, text_cols, datetime_cols):
        for t in ([primary] + (secondary or [])):
            if not t:
                continue
            for lst in (numeric_cols, categorical_cols, text_cols, datetime_cols):
                if t in lst:
                    lst.remove(t)
        return numeric_cols, categorical_cols, text_cols, datetime_cols

    numeric_cols, categorical_cols, text_cols, datetime_cols = remove_targets_from_lists(
        primary, secondary, numeric_cols, categorical_cols, text_cols, datetime_cols
    )

    # -------------------------------------
    # 2️⃣ Numeric Features (skip sentiment columns)
    # -------------------------------------
    def process_numeric(df, cols):
        scaler = StandardScaler()
        numeric_meta = {}
        for col in cols:
            if col.endswith(("_sentiment_num", "_sentiment_confidence")):
                continue  # skip derived sentiment numerics
            # coerce numeric and fill median
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().mean() > 0:
                df[col].fillna(df[col].median(), inplace=True)
            # create z-score column
            try:
                df[f"{col}_z"] = scaler.fit_transform(df[[col]])
            except Exception:
                # fallback: if scaler fails (constant), still create column of zeros
                df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() if df[col].std() != 0 else 1)
            numeric_meta[col] = {"mean": float(df[col].mean()), "std": float(df[col].std()), "filled": True}
        return df, numeric_meta

    df, numeric_meta = process_numeric(df, numeric_cols)
    print(f"✅ Processed {len(numeric_meta)} numeric columns (excluding sentiment).")

    # -------------------------------------
    # 3️⃣ Categorical Features (skip _sentiment columns)
    # -------------------------------------
    def process_categorical(df, cols):
        encoders = {}
        for col in cols:
            if col.endswith("_sentiment"):
                continue  # skip text-based sentiment columns
            s = df[col].astype(str).fillna("missing")
            if s.nunique() <= 30:
                le = LabelEncoder()
                try:
                    df[f"{col}_encoded"] = le.fit_transform(s)
                    mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
                except Exception:
                    # fallback: create frequency mapping if label encoding fails
                    freq = s.value_counts(normalize=True)
                    df[f"{col}_freq"] = s.map(freq)
                    mapping = None
                    encoders[col] = {"type": "freq", "mapping_size": len(freq)}
                    continue
                encoders[col] = {
                    "type": "label",
                    "mapping": mapping
                }
            else:
                freq = s.value_counts(normalize=True)
                df[f"{col}_freq"] = s.map(freq)
                encoders[col] = {"type": "freq", "mapping_size": len(freq)}
        return df, encoders

    df, encoders = process_categorical(df, categorical_cols)
    print(f"✅ Encoded {len(encoders)} categorical columns (excluding sentiment).")

    # -------------------------------------
    # 4️⃣ Text Features
    # -------------------------------------
    def text_features(df, text_cols):
        relations = {}
        for col in text_cols:
            s = df[col].astype(str).fillna("")
            df[f"{col}_word_count"] = s.apply(lambda x: len(x.split()))
            df[f"{col}_unique_ratio"] = s.apply(lambda x: len(set(x.split())) / max(1, len(x.split())))
            df[f"{col}_has_url"] = s.str.contains("http|www").astype(int)
            df[f"{col}_has_emoji"] = s.str.contains(r"[\U0001F600-\U0001F64F]", regex=True).astype(int)
            df[f"{col}_upper_ratio"] = s.apply(lambda x: sum(1 for c in x if c.isupper()) / max(1, len(x)))
            relations[col] = [
                f"{col}_word_count", f"{col}_unique_ratio", f"{col}_has_url",
                f"{col}_has_emoji", f"{col}_upper_ratio"
            ]
        return df, relations

    df, text_relations = text_features(df, text_cols)
    print(f"✅ Generated text metrics for {len(text_cols)} columns.")

    # -------------------------------------
    # 5️⃣ Sentiment Features (single aggregated metric)
    # -------------------------------------
    def sentiment_features(df):
        sent_cols = [c for c in df.columns if c.endswith("_sentiment_num")]
        conf_cols = [c for c in df.columns if c.endswith("_sentiment_confidence")]
        if not sent_cols:
            print("ℹ️ No sentiment columns found for aggregation.")
            return df, {}

        # single useful aggregation
        df["avg_sentiment_all_textcols"] = df[sent_cols].mean(axis=1)

        meta = {
            "sentiment_columns": sent_cols,
            "confidence_columns": conf_cols,
            "aggregations": ["avg_sentiment_all_textcols"]
        }
        print(f"🧠 Aggregated sentiment features from {len(sent_cols)} columns.")
        return df, meta

    df, sent_meta = sentiment_features(df)

    # -------------------------------------
    # 6️⃣ Datetime Features (profile-driven)
    # -------------------------------------
    def datetime_features(df, datetime_cols):
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                continue
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_is_weekend"] = (df[col].dt.weekday >= 5).astype(int)
        return df

    df = datetime_features(df, datetime_cols)
    print(f"🕒 Extracted datetime parts for {len(datetime_cols)} columns.")

    # -------------------------------------
    # 7️⃣ Row-level meta features
    # -------------------------------------
    df["n_missing"] = df.isna().sum(axis=1)
    df["n_text_cols_nonempty"] = df[text_cols].notna().sum(axis=1) if text_cols else 0
    df["has_urls"] = (
        df[[c for c in df.columns if "_has_url" in c]].max(axis=1)
        if any("_has_url" in c for c in df.columns) else 0
    )

    # -------------------------------------
    # 8️⃣ Cleanup: drop redundant or constant columns
    # Protect primary/secondary targets from being dropped even if constant
    # -------------------------------------
    protected_cols = [c for c in ([primary] + (secondary or [])) if c]
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c not in protected_cols]
    df.drop(columns=const_cols, inplace=True, errors="ignore")
    duplicate_cols = df.columns[df.T.duplicated()].tolist()
    if duplicate_cols:
        df.drop(columns=duplicate_cols, inplace=True, errors="ignore")

    print(f"🧹 Dropped {len(const_cols) + len(duplicate_cols)} redundant/constant columns.")

    # -------------------------------------
    # Build parent->derived and derived->parent mappings
    # -------------------------------------
    # Start from text_relations (text parent -> derived list)
    column_relations = dict(text_relations) if text_relations else {}

    # Add numeric _z fields
    for col in numeric_cols:
        zname = f"{col}_z"
        if zname in df.columns:
            column_relations.setdefault(col, []).append(zname)

    # Add categorical encodings / freq
    for col in categorical_cols:
        enc_label = f"{col}_encoded"
        freq_label = f"{col}_freq"
        if enc_label in df.columns:
            column_relations.setdefault(col, []).append(enc_label)
        if freq_label in df.columns:
            column_relations.setdefault(col, []).append(freq_label)

    # Add datetime derived parts
    for col in datetime_cols:
        for part in (f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_hour", f"{col}_weekday", f"{col}_is_weekend"):
            if part in df.columns:
                column_relations.setdefault(col, []).append(part)

    # Add sentiment-related columns that follow naming conventions
    # (parent_sentiment, parent_sentiment_num, parent_sentiment_confidence)
    for c in list(df.columns):
        if c.endswith("_sentiment"):
            parent = c[:-10]
            column_relations.setdefault(parent, []).append(c)
        if c.endswith("_sentiment_num"):
            parent = c[:-14]
            column_relations.setdefault(parent, []).append(c)
        if c.endswith("_sentiment_confidence"):
            parent = c[:-21]
            column_relations.setdefault(parent, []).append(c)

    # dedupe and keep only existing columns
    for p in list(column_relations.keys()):
        column_relations[p] = list(dict.fromkeys([x for x in column_relations.get(p, []) if x in df.columns]))

    # build derived_to_parent map (inverse)
    derived_to_parent = {}
    for parent, deriveds in column_relations.items():
        for d in deriveds:
            derived_to_parent[d] = parent

    # determine original_columns (prefer profile if present)
    if profile and "columns" in profile:
        original_columns = list(profile["columns"].keys())
    else:
        # best-effort: original columns from initial load
        original_columns = original_columns_on_load

    # compute feature_columns (engineered ones)
    feature_columns = [c for c in df.columns if c not in original_columns]

    # -------------------------------------
    # 9️⃣ Save results & metadata (rich)
    # -------------------------------------
    all_meta = {
        "numeric": numeric_meta,
        "categorical": encoders,
        "sentiment": sent_meta,
        "text_cols": text_cols,
        "datetime_cols": datetime_cols,
        "column_relations": column_relations,      # parent -> [derived columns]
        "derived_to_parent": derived_to_parent,    # derived -> parent
        "original_columns": original_columns,
        "primary_target": primary,
        "secondary_targets": secondary,
        "feature_columns": feature_columns
    }

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    print(f"✅ Saved enriched features → {out_csv}")
    print(f"🧾 Metadata saved → {out_meta}")
    print(f"Total columns after feature engineering: {df.shape[1]}")

    return df


# -------------------------------------
# CLI Entry Point
# -------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_engineer.py <config.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
