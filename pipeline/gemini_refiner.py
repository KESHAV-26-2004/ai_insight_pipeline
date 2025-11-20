# gemini_refiner.py
import os
import json
import textwrap
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Gemini SDK (import guarded)
try:
    import google.generativeai as genai  # pip install google-generativeai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

# ------------------------------
# Config
# ------------------------------
SAMPLES_PER_COLUMN = 6
REPORT_SUBDIR = "gemini"
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
GEMINI_API_KEY_ENV = "GOOGLE_API_KEY"

GEMINI_OUTPUT_FILENAME = "{base}_gemini_output.json"
REPORT_MD_FILENAME = "{base}_full_report.md"

# ------------------------------
# Helpers: file loaders
# ------------------------------
def load_json_safe(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_profile(profile_path: Path) -> Dict:
    return load_json_safe(profile_path)

def load_targets_meta(targets_meta_path: Path) -> Dict:
    return load_json_safe(targets_meta_path)

def load_feature_meta(feature_meta_path: Path) -> Dict:
    return load_json_safe(feature_meta_path)

def load_sentiment_meta(sentiment_meta_path: Path) -> Dict:
    return load_json_safe(sentiment_meta_path)

def load_relations(relations_path: Path) -> List[Dict]:
    data = load_json_safe(relations_path)
    if isinstance(data, list):
        return data
    return data.get("relations", []) if isinstance(data, dict) else []

# ------------------------------
# Prompt construction (as you specified)
# ------------------------------
def prettify_col_name(name: str) -> str:
    return name.replace("_", " ").strip().title()

def get_samples_for_column(colname: str, profile: Dict, df: pd.DataFrame = None, n: int = SAMPLES_PER_COLUMN) -> List[str]:
    cols = profile.get("columns", {}) if profile else {}
    if colname in cols:
        sv = cols[colname].get("sample_values", [])
        if sv:
            return [str(x) for x in sv[:n]]
    if df is not None and colname in df.columns:
        non_null = df[colname].dropna().astype(str)
        if len(non_null) == 0:
            return []
        # deterministic sample for reproducibility
        sample_count = min(n, max(1, len(non_null)))
        return non_null.sample(sample_count, random_state=1).tolist()
    return []

def build_prompt(
    title: str,
    targets_meta: Dict,
    profile: Dict,
    feature_meta: Dict,
    sentiment_meta: Dict,
    relations: List[Dict],
    features_df: pd.DataFrame
) -> str:
    primary = targets_meta.get("primary") or feature_meta.get("primary_target") or ""
    secondary = targets_meta.get("secondary") or feature_meta.get("secondary_targets") or []
    if isinstance(secondary, str):
        secondary = [secondary]
    n_rows = profile.get("n_rows") or (len(features_df) if features_df is not None else None)
    original_columns = feature_meta.get("original_columns") or list(profile.get("columns", {}).keys()) if profile else list(features_df.columns) if features_df is not None else []

    out_lines = []
    out_lines.append("You are an expert data analyst. I will give you a full dataset profile,")
    out_lines.append("sentiment analysis and top relations.")
    out_lines.append("")
    out_lines.append("Your job is to:")
    out_lines.append("1) Write a SINGLE final summary (6–10 sentences).")
    out_lines.append("2) Write detailed, actionable recommendations (5–12 items).")
    out_lines.append("")
    out_lines.append("---")
    out_lines.append(f"DATASET TITLE:\n{title or 'Unknown Title'}\n")
    out_lines.append("TASK:")
    out_lines.append(f"Primary Target: {primary or 'None'}")
    out_lines.append(f"Secondary Targets: {secondary or []}")
    out_lines.append("")
    out_lines.append(f"ROWS: {n_rows}")
    out_lines.append(f"COLUMNS (original): {', '.join(original_columns)}")
    out_lines.append("\n---\n")
    out_lines.append("COLUMN DETAILS (with 8 samples each):")
    df = features_df
    for col in original_columns:
        dtype = profile.get("columns", {}).get(col, {}).get("dtype") if profile else None
        dtype = dtype or (str(df[col].dtype) if (df is not None and col in df.columns) else "unknown")
        out_lines.append(f"{prettify_col_name(col)}: {dtype}")
        samples = get_samples_for_column(col, profile, df, n=SAMPLES_PER_COLUMN)
        if samples:
            out_lines.append("Samples:")
            for s in samples:
                s_clean = str(s).replace("\n", " ")
                out_lines.append(f"- {s_clean}")
        else:
            out_lines.append("- (no sample available)")
        out_lines.append("")

    out_lines.append("---")
    out_lines.append("SENTIMENT SUMMARY")
    sent_cols = feature_meta.get("sentiment", {}).get("sentiment_columns", []) or sentiment_meta.get("existing_sentiment", [])
    if isinstance(sent_cols, list) and sent_cols and isinstance(sent_cols[0], dict):
        sent_cols_list = [it.get("column") for it in sent_cols]
    else:
        sent_cols_list = sent_cols or []
    out_lines.append(f"Sentiment Columns: {sent_cols_list}")

    distribution_text = "Distribution not found"
    try:
        if features_df is not None and sent_cols_list:
            col_candidate = None
            for sc in sent_cols_list:
                if sc in features_df.columns:
                    col_candidate = sc
                    break
                numname = f"{sc}_num"
                if numname in features_df.columns:
                    col_candidate = numname
                    break
            if col_candidate:
                vc = features_df[col_candidate].value_counts(dropna=True).to_dict()
                distribution_text = ", ".join([f"{k}={v}" for k, v in vc.items()])
    except Exception:
        distribution_text = "Error computing distribution"
    out_lines.append(f"Distribution: {distribution_text}")
    out_lines.append("\n---\n")
    out_lines.append("RELATIONS (Parent-Level) — full list (sorted by rank if available):")
    if relations:
        for r in relations:
            pa = r.get("parent_a") or r.get("feature_a") or r.get("a")
            pb = r.get("parent_b") or r.get("feature_b") or r.get("b")
            eff = r.get("effect_size")
            direction = r.get("direction")
            method = r.get("method")
            sentence = r.get("short") or r.get("notes") or ""
            out_lines.append(f"- {prettify_col_name(pa)} ↔ {prettify_col_name(pb)} | effect={eff} | direction={direction} | method={method}")
            if sentence:
                out_lines.append(f"  Sentence: {sentence}")
    else:
        out_lines.append("- (no relations found)")

    out_lines.append("\n---\n")
    out_lines.append("Write:")
    out_lines.append("1) FINAL SUMMARY (must reflect dataset purpose + relations + sentiment).")
    out_lines.append("2) RECOMMENDATIONS (actionable, data-driven; 5–12 items).")
    out_lines.append("")
    out_lines.append("Be concise, do not repeat the same recommendation, and prefer high-precision actions.")
    out_lines.append("")
    return "\n".join(out_lines)

# ------------------------------
# Gemini integration
# ------------------------------
def check_gemini(verbose: bool = True) -> Dict[str, Any]:
    """
    Quick health-check for Gemini usage:
      - checks package import
      - checks GOOGLE_API_KEY env var
      - optionally issues a tiny test generate (safe, short)
    Returns dict with status keys: package_ok, api_key_present, can_call
    """
    out = {"package_ok": GEMINI_AVAILABLE, "api_key_present": False, "can_call": False, "error": None}
    key = os.environ.get(GEMINI_API_KEY_ENV)
    out["api_key_present"] = bool(key)
    if not GEMINI_AVAILABLE:
        out["error"] = "google.generativeai package not importable"
        if verbose:
            print("⚠ google.generativeai not installed (pip install google-generativeai).")
        return out
    if not key:
        out["error"] = f"{GEMINI_API_KEY_ENV} environment variable not set"
        if verbose:
            print(f"⚠ {GEMINI_API_KEY_ENV} not set. Export it first.")
        return out
    try:
        genai.configure(api_key=key)
        # minimal test call — short prompt
        model = genai.GenerativeModel(GEMINI_MODEL_NAME, generation_config=genai.GenerationConfig(max_output_tokens=16, temperature=0.0))
        resp = model.generate_content(contents=[{"role": "user", "parts": [{"text": "Say ok"}]}])
        raw = resp.candidates[0].content.parts[0].text.strip()
        out["can_call"] = bool(raw)
        if verbose:
            print("✅ Gemini test call succeeded.")
    except Exception as e:
        out["error"] = str(e)
        if verbose:
            print("❌ Gemini test call failed:", e)
    return out

def query_model_with_gemini(prompt: str) -> str:
    """
    Uses google.generativeai to generate text. Expects GOOGLE_API_KEY to be set.
    """
    key = os.environ.get(GEMINI_API_KEY_ENV)
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google.generativeai package not installed")
    if not key:
        raise RuntimeError(f"{GEMINI_API_KEY_ENV} not set")
    genai.configure(api_key=key)
    generation_config = genai.GenerationConfig(temperature=GEMINI_TEMPERATURE, max_output_tokens=GEMINI_MAX_TOKENS)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME, generation_config=generation_config)
    response = model.generate_content(contents=[{"role": "user", "parts": [{"text": prompt}]}])
    # extract text (defensive)
    raw_output = ""
    try:
        raw_output = response.candidates[0].content.parts[0].text.strip()
    except Exception:
        # Some SDK versions return different shapes; attempt best-effort
        try:
            raw_output = str(response)
        except Exception:
            raw_output = ""
    # cleanup JSON wrappers if any
    return raw_output

# ------------------------------
# Fallback / existing query_model
# ------------------------------
def query_model(prompt: str, timeout: int = 60) -> str:
    """
    Prefer Gemini when available + key present. Otherwise fallback to existing behavior:
      - try OpenAI if OPENAI_API_KEY set
      - else save prompt_debug.txt and read manual_response.txt
    """
    # try Gemini first
    key = os.environ.get(GEMINI_API_KEY_ENV)
    if GEMINI_AVAILABLE and key:
        try:
            return query_model_with_gemini(prompt)
        except Exception as e:
            # print and fallthrough to fallback (saves prompt)
            print("⚠ Gemini call failed, falling back. Error:", e)

    # try OpenAI as fallback (if user had previously used it)
    try:
        if os.getenv("OPENAI_API_KEY"):
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0
            )
            text = resp.choices[0].message.content
            return text
    except Exception:
        pass

    # fallback: save prompt for manual consumption
    debug_prompt_path = Path("prompt_debug.txt")
    debug_prompt_path.write_text(prompt, encoding="utf-8")
    manual_path = Path("manual_response.txt")
    if manual_path.exists():
        return manual_path.read_text(encoding="utf-8")

    placeholder = (
        "### MODEL_NOT_AVAILABLE_PLACEHOLDER\n\n"
        "No model API key was found or model call failed. The prompt has been saved to prompt_debug.txt.\n"
        "If you want to use Gemini, set the environment variable GOOGLE_API_KEY and install google-generativeai.\n"
        "Alternatively write the model response into 'manual_response.txt' and this script will pick it up.\n\n"
        "---- PROMPT START ----\n\n"
        + prompt[:2000]
        + "\n\n---- PROMPT END ----\n\n"
    )
    return placeholder

# ------------------------------
# Postprocess + save (unchanged)
# ------------------------------
def split_model_output(text: str) -> Dict[str, Any]:
    out = {"summary": "", "recommendations": []}
    if not text:
        return out
    txt = text.strip()
    lower = txt.lower()
    if "recommend" in lower:
        split_tokens = None
        for token in ["\nrecommendations", "\nrecommendation", "\n- recommendations", "\nRecommendations", "\nRecommendation"]:
            if token in txt:
                split_tokens = token
                break
        if split_tokens:
            parts = txt.split(split_tokens, maxsplit=1)
            out["summary"] = parts[0].strip()
            rec_part = parts[1].strip()
            recs = []
            for line in rec_part.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lstrip().startswith(("-", "*")):
                    recs.append(line.lstrip("-* ").strip())
                elif line and (line[0].isdigit() and (line[1] == "." or line[1] == ")")):
                    recs.append(line.split(".", 1)[1].strip() if "." in line else line)
                else:
                    recs.append(line)
            out["recommendations"] = recs
            return out
    sentences = txt.split(". ")
    if len(sentences) <= 4:
        out["summary"] = txt
        return out
    out["summary"] = ". ".join(sentences[:4]).strip()
    recs = []
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("- ") or line.startswith("* ") or (line and line[0].isdigit()):
            recs.append(line.lstrip("-*0123456789. )").strip())
    if not recs:
        rem = ". ".join(sentences[4:]).strip()
        if rem:
            recs = [s.strip() for s in rem.split("\n") if s.strip()]
    out["recommendations"] = recs
    return out

def save_outputs(base_folder: Path, base: str, model_text: str, split: Dict[str, Any]):
    # Save to: results/{dataset}/gemini/
    report_dir = base_folder / REPORT_SUBDIR
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    gemini_json_path = report_dir / GEMINI_OUTPUT_FILENAME.format(base=base)
    gemini_data = {
        "summary_raw": model_text,
        "summary": split.get("summary", ""),
        "recommendations": split.get("recommendations", [])
    }
    gemini_json_path.write_text(json.dumps(gemini_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown output
    md_path = report_dir / REPORT_MD_FILENAME.format(base=base)
    md_lines = []
    md_lines.append(f"# {base} — AI Summary & Recommendations\n")
    md_lines.append("## Final Summary\n")
    md_lines.append(split.get("summary", ""))

    md_lines.append("\n## Recommendations\n")
    for i, r in enumerate(split.get("recommendations", []), start=1):
        md_lines.append(f"{i}. {r}")

    md_text = "\n\n".join(md_lines)
    md_path.write_text(md_text, encoding="utf-8")

    return {
        "json": str(gemini_json_path.resolve()),
        "md": str(md_path.resolve())
    }

# ------------------------------
# Main entry
# ------------------------------
def generate_report_for_dataset(results_dataset_folder: str) -> Dict[str, Any]:
    base_folder = Path(results_dataset_folder)
    if not base_folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {base_folder}")
    base = base_folder.name
    profile_path = base_folder / "profiles" / f"{base}_profile.json"
    targets_meta_path = base_folder / "cleaned" / f"{base}_cleaned_targets_meta.json"
    sentiment_meta_path = base_folder / "enriched" / f"{base}_sentiment_meta.json"
    feature_meta_path = base_folder / "meta" / f"{base}_encoders.json"
    relations_path = base_folder / "relations" / f"{base}_relations.json"
    features_csv_path = base_folder / "features" / f"{base}_features.csv"

    profile = load_profile(profile_path)
    targets_meta = load_targets_meta(targets_meta_path)
    sentiment_meta = load_sentiment_meta(sentiment_meta_path)
    feature_meta = load_feature_meta(feature_meta_path)
    relations = load_relations(relations_path)
    df = None
    if features_csv_path.exists():
        try:
            df = pd.read_csv(features_csv_path, low_memory=False)
        except Exception:
            df = None

    # Load AI-generated title
    title_path = base_folder / "title" / f"{base}_title.json"
    title_json = load_json_safe(title_path)

    ai_title = title_json.get("generated_title", base)
    csv_name = f"{base}.csv"

    # Combine both titles for clarity
    full_title = f"{ai_title} (Source: {csv_name})"

    prompt = build_prompt(
        title=full_title,
        targets_meta=targets_meta,
        profile=profile,
        feature_meta=feature_meta,
        sentiment_meta=sentiment_meta,
        relations=relations,
        features_df=df
    )

    model_text = query_model(prompt)
    parsed = split_model_output(model_text)
    saved = save_outputs(base_folder, base, model_text, parsed)
    return {
        "prompt_file": str(Path("prompt_debug.txt").resolve()) if Path("prompt_debug.txt").exists() else None,
        "model_text_snippet": (model_text or "")[:1000],
        "saved_paths": saved
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Gemini-powered recommendations + summary from pipeline outputs.")
    parser.add_argument("--dataset_folder", required=True, help="Path to results/{dataset} folder (e.g. ./results/Genshin Impact_cleaned_with_sentiment)")
    parser.add_argument("--check_gemini", action="store_true", help="Perform a lightweight Gemini SDK + key check and exit.")
    args = parser.parse_args()

    if args.check_gemini:
        print("Running Gemini health-check...")
        print(check_gemini())
        raise SystemExit(0)

    res = generate_report_for_dataset(args.dataset_folder)
    print("Done. Outputs:", res)
