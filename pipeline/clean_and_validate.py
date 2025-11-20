import os, json, re
import pandas as pd
import numpy as np
from collections import Counter

# Defaults
DEFAULT_PROFILE_PATH = "../results/profiles"
DEFAULT_RESULTS_CLEANED = "../results/cleaned"

def read_csv_safe(path):
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    raise ValueError(f"Unable to read CSV: {path}")

def is_probable_id_column(series, col_name):
    """Return True if column is likely an ID, else False."""
    s = series.dropna().astype(str)
    if s.empty:
        return False

    if pd.api.types.is_numeric_dtype(series):
        if s.nunique() / len(s) > 0.95:
            return True
        return False

    name_score = 1 if re.search(r"(id|uuid|guid|hash|code|key)$", col_name.lower()) else 0

    lens = s.map(len)
    len_std = lens.std()
    len_mean = lens.mean()

    uniform_len_score = 1 if len_std < 2 else 0
    alphanum_frac = np.mean(s.str.match(r"^[A-Za-z0-9_-]+$"))
    alphanum_score = 1 if alphanum_frac > 0.7 else 0
    length_range_score = 1 if 8 <= len_mean <= 40 else 0

    sample = " ".join(s.sample(min(len(s), 200), random_state=42))
    has_words = bool(re.search(
        r"\b(good|bad|game|love|hate|amazing|excellent|great|poor|review|experience|fun|graphics|hotel|service)\b",
        sample.lower()
    ))
    space_frac = np.mean(s.str.contains(r"\s"))
    looks_like_text = has_words or space_frac > 0.05

    score = name_score + uniform_len_score + alphanum_score + length_range_score
    return score >= 3 and not looks_like_text


def clean_and_validate(profile_path,
                       csv_path,
                       primary_target,
                       secondary_targets,
                       task_type,
                       out_dir):
    """
    Cleans the dataset, validates user-selected targets,
    and saves cleaned CSV + metadata.
    """
    # Load profile
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    base = profile["file"]

    # Resolve CSV
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"📂 Using input CSV → {csv_path}")

    # Load CSV
    df = read_csv_safe(csv_path)
    print(f"📄 Loaded {df.shape[0]} rows, {df.shape[1]} cols")

    # ----------------------------------------------------
    # 1️⃣ Type Correction (same as old logic)
    # ----------------------------------------------------
    profile_changed = False
    for col, info in profile["columns"].items():
        if col not in df.columns:
            continue

        if info.get("dtype") == "datetime":
            s = df[col].astype(str)
            numeric_like = s.str.match(r"^[0-9]+(\.[0-9]+)?$").mean() > 0.8
            if numeric_like:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                profile["columns"][col]["dtype"] = "numeric"
                profile_changed = True
                print(f"🔧 Fixed '{col}' → numeric (was datetime)")

        elif info.get("dtype") == "numeric":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.95:
                    df[col] = parsed
                    profile["columns"][col]["dtype"] = "datetime"
                    profile_changed = True
                    print(f"🔧 Fixed '{col}' → datetime (was numeric)")
            except Exception:
                pass

    if profile_changed:
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        print(f"✅ Profile updated → {profile_path}")

    # ----------------------------------------------------
    # 2️⃣ Drop unwanted columns (same as old logic)
    # ----------------------------------------------------
    drop_cols, keep_datetime_cols = [], []

    for col in df.columns:
        s = df[col]
        name = col.lower()

        if s.isna().all() or s.nunique(dropna=True) <= 1:
            drop_cols.append(col)
            continue

        if is_probable_id_column(s, name):
            drop_cols.append(col)
            continue

        if re.search(r"(username|user_name|author|profile|player)", name):
            drop_cols.append(col)
            continue

        if (not pd.api.types.is_numeric_dtype(s)) and re.search(r"(timestamp|date|time)$", name):
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                parsed_frac = parsed.notna().mean()
                unique_days = parsed.dt.date.nunique()
                if parsed_frac > 0.8 and unique_days <= 1:
                    drop_cols.append(col)
                elif parsed_frac > 0.8:
                    keep_datetime_cols.append(col)
                    df[col] = parsed
            except Exception:
                pass

    df.drop(columns=drop_cols, inplace=True)
    print(f"🗑️ Dropped columns → {drop_cols}")

    # ----------------------------------------------------
    # 3️⃣ Build targets meta (USING USER INPUT)
    # ----------------------------------------------------

    # Ensure secondary is list
    if isinstance(secondary_targets, str):
        secondary_targets = [secondary_targets]
    secondary_targets = secondary_targets[:2]   # max 2

    targets_meta = {
        "file": f"{base}.csv",
        "primary": primary_target,
        "secondary": secondary_targets,
        "type": task_type,     # classification or regression
        "auto_changes": [],
        "needs_review": False
    }

    # Validate primary exists
    if primary_target not in df.columns:
        print(f"❌ Primary target '{primary_target}' not found!")
        targets_meta["needs_review"] = True
    else:
        col = df[primary_target]
        n_classes = col.nunique(dropna=True)

        if task_type == "classification":
            if not pd.api.types.is_numeric_dtype(col):
                # keep as it is (text classes allowed)
                pass
            if n_classes > 50:
                top = [x for x, _ in Counter(col).most_common(10)]
                df[primary_target] = df[primary_target].apply(lambda x: x if x in top else "__OTHER__")
                targets_meta["auto_changes"].append("collapsed to top 10 classes")
                targets_meta["needs_review"] = True

        elif task_type == "regression":
            if not pd.api.types.is_numeric_dtype(col):
                mapping = {v: i for i, v in enumerate(col.dropna().unique())}
                df[primary_target] = df[primary_target].map(mapping)
                targets_meta["auto_changes"].append("label-encoded to numeric")
                targets_meta["needs_review"] = True

    # ----------------------------------------------------
    # 4️⃣ Save cleaned CSV + meta
    # ----------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    clean_csv = f"{out_dir}/{base}_cleaned.csv"
    df.to_csv(clean_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Cleaned CSV → {clean_csv}")

    meta_path = f"{out_dir}/{base}_cleaned_targets_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(targets_meta, f, indent=2)

    print(f"✅ Targets Meta Saved → {meta_path}")

    return df, targets_meta, profile
