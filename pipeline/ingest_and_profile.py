# ingest_and_profile.py

import os, json, re
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

def read_csv_safe(path):
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    raise ValueError(f"Unable to read CSV: {path}")

def profile_csv(csv_path, out_dir="../results/profiles", sample_size=500):
    """
    Profiles csv_path and writes a profile JSON to out_dir.
    Returns (df, profile_dict)
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(csv_path).replace(".csv","")

    df = read_csv_safe(csv_path)
    print(f"📄 Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    profile = {"file": base, "n_rows": len(df), "columns": {}}

    for col in tqdm(df.columns):
        s = df[col]
        non_null = s.dropna()
        dtype = "unknown"

        # detect datetime
        if pd.api.types.is_datetime64_any_dtype(s):
            dtype = "datetime"
        else:
            try:
                dt_parsed = pd.to_datetime(non_null.sample(min(sample_size,len(non_null))), errors="coerce")
                if dt_parsed.notna().mean() > 0.95:
                    dtype = "datetime"
            except: pass

        if dtype != "datetime":
            if pd.api.types.is_numeric_dtype(s):
                dtype = "numeric"
            else:
                nunique = non_null.nunique()
                uniq_frac = nunique / len(non_null) if len(non_null)>0 else 0
                avg_len = non_null.astype(str).map(len).mean() if len(non_null)>0 else 0
                if uniq_frac < 0.05 or nunique <= 30:
                    dtype = "categorical"
                elif avg_len > 20:
                    dtype = "text"
                else:
                    dtype = "categorical"

        profile["columns"][col] = {
            "dtype": dtype,
            "non_null": int(non_null.shape[0]),
            "unique": int(non_null.nunique()),
            "sample_values": non_null.astype(str).head(5).tolist()
        }

    out_path = os.path.join(out_dir, f"{base}_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"✅ Profile saved → {out_path}")
    return df, profile

# No top-level execution: import-safe.
