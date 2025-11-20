# pipeline/relation_analyzer.py
"""
Relation & Visualization Analyzer (parent-level, quality-first).

See conversation with Keshav — this file:
- reads feature CSV + profile + cleaned targets meta + sentiment meta + feature meta
- selects representative derived columns for each original parent
- computes pairwise relations using appropriate tests
- collapses relations to parent-pairs choosing the best relation per parent pair
- applies FDR correction and strong quality filters
- generates a small number of high-quality plots (labels use prettified parent names)
- saves JSONs and summary

Outputs:
- results/{dataset}/relations/{dataset}_relations.json
- results/{dataset}/relations/{dataset}_relations_sentences.json
- results/{dataset}/plots/relation_1.png ...
- results/{dataset}/relations/summary.txt
"""
import os
import json
import math
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# ---------------------------
# Statistical helpers
# ---------------------------

def safe_pearson(x: np.ndarray, y: np.ndarray):
    try:
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None, None
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return None, None

def safe_spearman(x: np.ndarray, y: np.ndarray):
    try:
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return None, None
        rho, p = spearmanr(x, y, nan_policy='omit')
        return float(rho), float(p)
    except Exception:
        return None, None

def safe_chi2_and_cramers_v(a: pd.Series, b: pd.Series):
    try:
        ct = pd.crosstab(a.fillna("__MISSING__"), b.fillna("__MISSING__"))
        if ct.size == 0:
            return None, None
        chi2, p, dof, expected = chi2_contingency(ct, correction=False)
        n = ct.to_numpy().sum()
        r, k = ct.shape
        denom = n * (min(r, k) - 1) if (min(r, k) - 1) > 0 else 1
        cramers_v = math.sqrt(chi2 / denom) if denom > 0 else 0.0
        return float(cramers_v), float(p)
    except Exception:
        return None, None

def safe_anova_eta2(df: pd.DataFrame, numeric_col: str, cat_col: str):
    try:
        sub = df[[numeric_col, cat_col]].dropna()
        if sub.shape[0] < 10:
            return None, None
        tmp = sub.rename(columns={numeric_col: "y", cat_col: "g"})
        if tmp["g"].nunique() < 2:
            return None, None
        model = ols('y ~ C(g)', data=tmp).fit()
        anova_res = anova_lm(model, typ=2)
        ss_between = anova_res.loc['C(g)', 'sum_sq'] if 'C(g)' in anova_res.index else anova_res['sum_sq'].iloc[0]
        ss_total = ((tmp['y'] - tmp['y'].mean()) ** 2).sum()
        eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
        p = float(anova_res.loc['C(g)', 'PR(>F)']) if 'C(g)' in anova_res.index else float(anova_res['PR(>F)'].iloc[0])
        return eta2, p
    except Exception:
        return None, None

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json_safe(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_constant_series(s: pd.Series):
    return s.nunique(dropna=False) <= 1

def prettify(col: str) -> str:
    """Replace underscores with spaces and Title Case for display."""
    return col.replace("_", " ").strip().title()

# ---------------------------
# Derived/parent mapping helpers
# ---------------------------

def build_derived_to_parent_map(feature_meta: dict, sentiment_meta: dict) -> dict:
    if not feature_meta:
        feature_meta = {}
    d2p = {}
    if 'derived_to_parent' in feature_meta:
        d2p = dict(feature_meta['derived_to_parent'])
    else:
        # invert column_relations
        for parent, derived in feature_meta.get('column_relations', {}).items():
            for d in derived:
                d2p[d] = parent
    # include sentiment existing_sentiment mapping (if not present)
    for s in sentiment_meta.get('existing_sentiment', []) if sentiment_meta else []:
        parent = s.get('column')
        num = s.get('numeric_version')
        if num:
            d2p.setdefault(num, parent)
            d2p.setdefault(parent + "_sentiment", parent)
    return d2p

def representative_for_parent(parent: str, df: pd.DataFrame, feature_meta: dict, sentiment_meta: dict):
    """
    Preference:
    1) parent_sentiment_num
    2) numeric derived (z, _num, _confidence)
    3) encoded/freq numeric derived
    4) datetime-derived if parent is datetime (hour/day)
    5) fallback to parent itself
    """
    # 1
    cand = f"{parent}_sentiment_num"
    if cand in df.columns:
        return cand
    # from column_relations
    rels = feature_meta.get('column_relations', {}) if feature_meta else {}
    for d in rels.get(parent, []):
        # choose numeric-like deriveds first
        if d in df.columns and pd.api.types.is_numeric_dtype(df[d]):
            return d
        # also accept names ending in _num/_z/_confidence/_freq/_encoded
        if d in df.columns and any(d.endswith(s) for s in ("_num", "_z", "_confidence", "_freq", "_encoded")):
            return d
    # datetime-derived preference
    for p in (f"{parent}_hour", f"{parent}_day", f"{parent}_month", f"{parent}_year"):
        if p in df.columns and pd.api.types.is_numeric_dtype(df[p]):
            return p
    # fallback to parent
    if parent in df.columns:
        return parent
    # last resort: any derived if exists
    if rels.get(parent):
        for d in rels.get(parent):
            if d in df.columns:
                return d
    return parent

# ---------------------------
# Pair building -> parent-level
# ---------------------------

def build_parent_pairs(df: pd.DataFrame, profile: dict, targets_meta: dict, sentiment_meta: dict, feature_meta: dict):
    """
    Build parent pairs (parent_a, parent_b) and choose representatives (rep_a, rep_b)
    Returns list of tuples: (parent_a, parent_b, rep_a, rep_b)
    """
    d2p = build_derived_to_parent_map(feature_meta, sentiment_meta)
    # parents = all unique parents for columns present in df
    parents_set = set()
    for c in df.columns:
        parents_set.add(d2p.get(c, c))
    parents = sorted(list(parents_set))

    primary = targets_meta.get("primary")
    secondary = targets_meta.get("secondary") or []
    # normalize secondary list
    if isinstance(secondary, str):
        secondary = [secondary]

    pairs = []
    seen_parents = set()

    def add_parent_pair(pa, pb):
        if pa == pb:
            return
        key = tuple(sorted((pa, pb)))
        if key in seen_parents:
            return
        # block if one is parent of the other (don't pair parent with its own derived) — handled since pa,pb are parents
        # choose representatives
        rep_a = representative_for_parent(pa, df, feature_meta, sentiment_meta)
        rep_b = representative_for_parent(pb, df, feature_meta, sentiment_meta)
        # require both reps present and not identical
        if rep_a not in df.columns or rep_b not in df.columns or rep_a == rep_b:
            return
        # skip if no variation
        if is_constant_series(df[rep_a]) or is_constant_series(df[rep_b]):
            return
        seen_parents.add(key)
        pairs.append((pa, pb, rep_a, rep_b))

    # Primary vs all parents
    if primary and primary in parents:
        for p in parents:
            if p == primary:
                continue
            add_parent_pair(primary, p)

    # Secondary vs others
    for s in secondary:
        if s not in parents:
            continue
        for p in parents:
            if p == s or p == primary:
                continue
            add_parent_pair(s, p)

    # top numeric parents interactions (small set)
    numeric_parents = [p for p in parents if profile.get("columns", {}).get(p, {}).get("dtype") == "numeric"]
    try:
        if numeric_parents:
            reps = [representative_for_parent(p, df, feature_meta, sentiment_meta) for p in numeric_parents]
            # filter present numeric reps
            reps_exist = [r for r in reps if r in df.columns]
            if reps_exist:
                stds = df[reps_exist].std(numeric_only=True, skipna=True).sort_values(ascending=False)
                top_reps = list(stds.head(6).index)
                # map back to parent via d2p
                top_parents = [d2p.get(r, r) for r in top_reps]
                for i in range(len(top_parents)):
                    for j in range(i+1, len(top_parents)):
                        add_parent_pair(top_parents[i], top_parents[j])
    except Exception:
        pass

    return pairs

# ---------------------------
# Relation computation (on representative columns)
# ---------------------------

def compute_relation_on_reps(df: pd.DataFrame, parent_a: str, parent_b: str, rep_a: str, rep_b: str, profile: dict):
    """
    Compute relation metrics on rep_a and rep_b. Returns dict including parent names and reps.
    """
    # determine types using profile if available for parents else fallback to series types
    cols_meta = profile.get("columns", {}) if profile else {}
    # try to infer type of rep columns since reps may be derived
    def effective_type_series(col_name):
        s = df[col_name]
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            nn = s.dropna()
            if nn.shape[0] == 0:
                return "categorical"
            avg_words = np.mean(nn.astype(str).map(lambda x: len(x.split())))
            if avg_words < 3 and nn.nunique() <= 50:
                return "categorical"
            return "text"
        return "categorical"

    t_a = effective_type_series(rep_a)
    t_b = effective_type_series(rep_b)

    res = {
        "parent_a": parent_a,
        "parent_b": parent_b,
        "feature_a": rep_a,
        "feature_b": rep_b,
        "method": None,
        "effect_size": None,
        "p_value": None,
        "n": int(df[[rep_a, rep_b]].dropna().shape[0]),
        "direction": None,
        "notes": None
    }

    n_valid = df[[rep_a, rep_b]].dropna().shape[0]
    if n_valid < 10:
        res["notes"] = "too_few_samples"
        return res

    # numeric-numeric
    if t_a == "numeric" and t_b == "numeric":
        x = pd.to_numeric(df[rep_a], errors="coerce")
        y = pd.to_numeric(df[rep_b], errors="coerce")
        pear_r, pear_p = safe_pearson(x.dropna().values, y.dropna().values)
        spe_rho, spe_p = safe_spearman(x.values, y.values)
        method = "spearman" if spe_rho is not None else "pearson"
        effect = spe_rho if spe_rho is not None else pear_r
        pval = spe_p if spe_rho is not None else pear_p
        if effect is None:
            res["notes"] = "constant_or_error"
            return res
        res.update({"method": method, "effect_size": float(effect), "p_value": float(pval), "direction": "positive" if effect > 0 else "negative"})
        return res

    # numeric vs categorical/text (ANOVA)
    if (t_a == "numeric" and t_b in ("categorical", "text")) or (t_b == "numeric" and t_a in ("categorical", "text")):
        if t_b == "numeric":
            num_col, cat_col = rep_b, rep_a
        else:
            num_col, cat_col = rep_a, rep_b
        eta2, p = safe_anova_eta2(df, num_col, cat_col)
        if eta2 is None:
            res["notes"] = "anova_failed_or_insufficient"
            return res
        means = df.groupby(cat_col)[num_col].mean().dropna()
        if means.shape[0] >= 2:
            top, bottom = means.max(), means.min()
            direction = "positive" if top > bottom else "negative"
        else:
            direction = None
        res.update({"method": "anova_eta2", "effect_size": float(eta2), "p_value": float(p), "direction": direction})
        return res

    # categorical vs categorical
    if t_a in ("categorical", "text") and t_b in ("categorical", "text"):
        v, p = safe_chi2_and_cramers_v(df[rep_a], df[rep_b])
        if v is None:
            res["notes"] = "chi2_failed_or_insufficient"
            return res
        res.update({"method": "cramers_v", "effect_size": float(v), "p_value": float(p), "direction": None})
        return res

    # numeric vs datetime (treat hour)
    if (t_a == "numeric" and t_b == "datetime") or (t_b == "numeric" and t_a == "datetime"):
        if t_a == "numeric":
            num_col, dt_col = rep_a, rep_b
        else:
            num_col, dt_col = rep_b, rep_a
        try:
            dt = pd.to_datetime(df[dt_col], errors="coerce")
            hour = dt.dt.hour
            num = pd.to_numeric(df[num_col], errors="coerce")
            if hour.dropna().shape[0] < 10:
                res["notes"] = "datetime_insufficient"
                return res
            rho, p = safe_spearman(num.values, hour.values)
            if rho is None:
                res["notes"] = "constant_or_error"
                return res
            res.update({"method": "spearman_time_hour", "effect_size": float(rho), "p_value": float(p), "direction": "positive" if rho > 0 else "negative"})
            return res
        except Exception:
            res["notes"] = "datetime_error"
            return res

    # fallback to chi2
    v, p = safe_chi2_and_cramers_v(df[rep_a], df[rep_b])
    if v is None:
        res["notes"] = "fallback_failed"
        return res
    res.update({"method": "cramers_v", "effect_size": float(v), "p_value": float(p), "direction": None})
    return res

# ---------------------------
# Sentence generation & plotting (parent labels)
# ---------------------------

def strength_label(effect):
    if effect is None:
        return "no-result"
    ae = abs(effect)
    if ae > 0.3:
        return "strong"
    if ae > 0.1:
        return "moderate"
    return "weak"

def relation_to_sentence_parent(r: dict) -> dict:
    a = prettify(r.get("parent_a", r.get("feature_a")))
    b = prettify(r.get("parent_b", r.get("feature_b")))
    method = r.get("method")
    effect = r.get("effect_size")
    p = r.get("p_value")
    direction = r.get("direction")
    n = r.get("n", None)
    if effect is None:
        return {"short": f"No clear relation found between {a} and {b}.", "long": f"No statistically meaningful relation was detected between {a} and {b}.", "action": ""}
    strength = strength_label(effect)
    dir_word = "increases with" if direction == "positive" else ("decreases with" if direction == "negative" else "is associated with")
    short = f"{a} {dir_word} {b} ({strength}, {method}, effect={effect:.2f}, p={p:.3g})."
    long = f"A {strength} relationship was found between {a} and {b} using {method} (effect={effect:.3f}, p={p:.3g}, n={n}). This indicates that {a} {dir_word} {b}."
    if strength == "strong":
        action = f"Consider prioritizing {a} and {b} for interventions or further testing."
    elif strength == "moderate":
        action = f"Investigate {a} vs {b} further — they show a moderate relationship."
    else:
        action = f"Relationship weak — likely not a primary driver."
    return {"short": short, "long": long, "action": action}

def plot_parent_relation(df: pd.DataFrame, rel: dict, out_path: str):
    """
    Plot using representative columns but label with parent names (prettified).
    rel contains: parent_a,parent_b,feature_a(rep),feature_b(rep),method,effect_size...
    """
    rep_a = rel.get("feature_a")
    rep_b = rel.get("feature_b")
    pa = prettify(rel.get("parent_a"))
    pb = prettify(rel.get("parent_b"))
    method = rel.get("method", "")
    ensure_dir(os.path.dirname(out_path))
    try:
        s_a = df[rep_a]
        s_b = df[rep_b]

        def t_of(s):
            if pd.api.types.is_numeric_dtype(s): return "numeric"
            if pd.api.types.is_datetime64_any_dtype(s): return "datetime"
            nn = s.dropna()
            if nn.empty:
                return "categorical"
            avg_words = np.mean(nn.astype(str).map(lambda x: len(x.split())))
            if avg_words < 3 and nn.nunique() <= 50:
                return "categorical"
            return "text"

        ta = t_of(s_a)
        tb = t_of(s_b)

        plt.figure(figsize=(6, 4))
        title = f"{pa} vs {pb} — {method} (effect={rel.get('effect_size', 0):.2f})"
        plt.title(title)

        if ta == "numeric" and tb == "numeric":
            x = pd.to_numeric(s_a, errors="coerce")
            y = pd.to_numeric(s_b, errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() == 0:
                plt.text(0.5, 0.5, "No data", ha="center")
            else:
                x_ = x[ok]
                y_ = y[ok]
                plt.scatter(x_, y_, s=6, alpha=0.5)
                try:
                    coeffs = np.polyfit(x_, y_, deg=1)
                    xs = np.linspace(x_.min(), x_.max(), 100)
                    ys = np.polyval(coeffs, xs)
                    plt.plot(xs, ys, linewidth=1.5)
                except Exception:
                    pass
                plt.xlabel(pa)
                plt.ylabel(pb)

        elif (ta == "numeric" and tb in ("categorical", "text")) or (tb == "numeric" and ta in ("categorical", "text")):
            if ta == "numeric":
                num = pd.to_numeric(s_a, errors="coerce")
                cat = s_b.fillna("__MISSING__").astype(str)
                groups = []
                labels = []
                for name, g in pd.concat([num, cat], axis=1).groupby(cat):
                    vals = g[num.name].dropna()
                    if len(vals) > 0:
                        groups.append(vals.values)
                        labels.append(str(name)[:30])
                if groups:
                    plt.boxplot(groups, labels=labels, showfliers=False)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel(pa)
                    plt.xlabel(pb)
                else:
                    plt.text(0.5, 0.5, "No groups with numeric data", ha="center")
            else:
                num = pd.to_numeric(s_b, errors="coerce")
                cat = s_a.fillna("__MISSING__").astype(str)
                groups = []
                labels = []
                for name, g in pd.concat([num, cat], axis=1).groupby(cat):
                    vals = g[num.name].dropna()
                    if len(vals) > 0:
                        groups.append(vals.values)
                        labels.append(str(name)[:30])
                if groups:
                    plt.boxplot(groups, labels=labels, showfliers=False)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel(pb)
                    plt.xlabel(pa)
                else:
                    plt.text(0.5, 0.5, "No groups with numeric data", ha="center")

        elif ta in ("categorical", "text") and tb in ("categorical", "text"):
            ct = pd.crosstab(s_a.fillna("__MISSING__").astype(str), s_b.fillna("__MISSING__").astype(str))
            if ct.size == 0:
                plt.text(0.5, 0.5, "No cross-tabs", ha="center")
            else:
                # limit size for readability
                max_dim = 30
                ct_plot = ct.copy()
                if ct_plot.shape[0] > max_dim:
                    ct_plot = ct_plot.iloc[:max_dim, :]
                if ct_plot.shape[1] > max_dim:
                    ct_plot = ct_plot.iloc[:, :max_dim]
                plt.imshow(ct_plot.values, aspect='auto', interpolation='nearest')
                plt.colorbar()
                plt.xticks(range(len(ct_plot.columns)), [str(x)[:20] for x in ct_plot.columns], rotation=45, ha='right')
                plt.yticks(range(len(ct_plot.index)), [str(x)[:20] for x in ct_plot.index])
                plt.xlabel(pb)
                plt.ylabel(pa)

        elif ta == "numeric" and tb == "datetime" or ta == "datetime" and tb == "numeric":
            if ta == "datetime":
                dt = pd.to_datetime(s_a, errors="coerce")
                num = pd.to_numeric(s_b, errors="coerce")
            else:
                dt = pd.to_datetime(s_b, errors="coerce")
                num = pd.to_numeric(s_a, errors="coerce")
            tmp = pd.concat([dt, num], axis=1).dropna()
            if tmp.shape[0] == 0:
                plt.text(0.5, 0.5, "No data for trend", ha="center")
            else:
                tmp.index = pd.to_datetime(tmp.iloc[:, 0])
                agg = tmp.iloc[:, 1].resample('D').mean()
                plt.plot(agg.index, agg.values, linewidth=1.5)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('mean')
        else:
            plt.text(0.5, 0.5, "Plot not supported for types", ha="center")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

# ---------------------------
# Main analyze_relations entry
# ---------------------------

def analyze_relations(
    dataset_name: str,
    features_path: str,
    profile_path: str,
    targets_meta_path: str,
    sentiment_meta_path: str,
    feature_meta_path: str,
    out_dir: str = None,
    pval_threshold: float = 0.05,
    effect_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Main entry.
    """
    if out_dir is None:
        base_folder = os.path.dirname(os.path.dirname(features_path))
        out_dir = os.path.join(base_folder, "relations")
    plots_dir = os.path.join(os.path.dirname(out_dir), "plots")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    df = pd.read_csv(features_path, low_memory=False)

    profile = load_json_safe(profile_path)
    targets_meta = load_json_safe(targets_meta_path)
    sentiment_meta = load_json_safe(sentiment_meta_path)
    feature_meta = load_json_safe(feature_meta_path)

    # ====================== DEBUG BLOCK ======================
    print("\n================ DEBUG: Relation Analyzer Input Check ================")
    print(f"➡ features_path argument: {features_path}")

    # Check if file exists
    if os.path.exists(features_path):
        print("✔ File exists ✔")
    else:
        print("❌ ERROR: features_path NOT FOUND ❌")

    # Print directory contents
    dir_path = os.path.dirname(features_path)
    print(f"\n📂 Directory containing file: {dir_path}")
    print("📁 Files in this directory:")
    try:
        print(os.listdir(dir_path))
    except Exception as e:
        print(f"❌ Cannot list directory: {e}")

    # Print first few rows and columns
    print("\n🔍 CSV Columns Loaded:")
    print(df.columns.tolist())

    print("\n🔍 Checking key sentiment columns:")
    for col in ["content_sentiment", "content_sentiment_num", "content_sentiment_confidence"]:
        if col in df.columns:
            print(f"✔ FOUND: {col}")
        else:
            print(f"❌ MISSING: {col}")

    print("\n🔍 Checking text parent 'content' derived columns (column_relations):")
    try:
        content_rels = feature_meta.get("column_relations", {}).get("content", [])
        print("From feature_meta:", content_rels)
        for c in content_rels:
            exists = c in df.columns
            print(f"  {c}: {'✔ Present' if exists else '❌ Missing'}")
    except:
        print("⚠ No feature_meta or no column_relations available.")

    print("================ END DEBUG ========================================\n")
    # ================================================================

    # Build parent pairs with representatives
    parent_pairs = build_parent_pairs(df, profile, targets_meta, sentiment_meta, feature_meta)
    print(f"🔢 Candidate parent pairs (with reps): {len(parent_pairs)}")

    # Compute stats on reps
    raw_results = []
    for pa, pb, rep_a, rep_b in parent_pairs:
        # skip pairing parent with its own derived representative (shouldn't happen because pa!=pb)
        r = compute_relation_on_reps(df, pa, pb, rep_a, rep_b, profile)
        if r.get("notes"):
            # skip immediate failures
            continue
        # attach priority weight: prefer primary, then secondary
        weight = 0
        primary = targets_meta.get("primary")
        secondary = targets_meta.get("secondary") or []
        if isinstance(secondary, str):
            secondary = [secondary]
        if primary in (pa, pb):
            weight += 3
        if any(s in (pa, pb) for s in secondary):
            weight += 2
        r["priority_weight"] = weight
        raw_results.append(r)

    # Collapse to parent-level: choose the best derived rep pair per parent pair
    collapsed = {}
    for r in raw_results:
        key = tuple(sorted((r["parent_a"], r["parent_b"])))
        eff = abs(r.get("effect_size") or 0)
        pval = r.get("p_value") if r.get("p_value") is not None else 1.0
        score = eff * (1 + r.get("priority_weight", 0)) * (1 - min(1.0, pval))
        # prefer higher n if tie
        if key not in collapsed or (score > collapsed[key]["_score"]):
            r_copy = dict(r)
            r_copy["_score"] = score
            collapsed[key] = r_copy

    parent_results = list(collapsed.values())
    if not parent_results:
        # no viable relations found
        print("ℹ️ No viable parent-level relations found.")
        # save empty outputs
        relations_json_path = os.path.join(out_dir, f"{dataset_name}_relations.json")
        sentences_json_path = os.path.join(out_dir, f"{dataset_name}_relations_sentences.json")
        with open(relations_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        with open(sentences_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return {"relations_path": relations_json_path, "sentences_path": sentences_json_path, "plots_dir": plots_dir}

    # Multiple test correction on parent-level p-values
    pvals = [r["p_value"] for r in parent_results]
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    for i, r in enumerate(parent_results):
        r["adjusted_p_value"] = float(pvals_corrected[i])
        r["significant_fdr"] = bool(reject[i])

    # Ranking score
    for r in parent_results:
        eff = abs(r.get("effect_size") or 0)
        adj_p = r.get("adjusted_p_value") if r.get("adjusted_p_value") is not None else r.get("p_value", 1.0)
        p_penalty = (1 - min(1.0, adj_p))
        r["rank_score"] = eff * (1 + r.get("priority_weight", 0)) * p_penalty

    # Strict quality filtering
    # - effect >= effect_threshold
    # - adjusted p <= pval_threshold
    # - for cat-cat ensure Cramer's V >= effect_threshold
    total_n = df.shape[0]
    def min_points_required(n_rows):
        if n_rows < 100:
            return max(10, int(n_rows * 0.1))
        return 50

    min_points = min_points_required(total_n)

    qual = []
    for r in parent_results:
        if r.get("n", 0) < min_points:
            continue
        if not r.get("significant_fdr", False):
            continue
        if r.get("method") == "cramers_v" and abs(r.get("effect_size", 0)) < effect_threshold:
            continue
        if abs(r.get("effect_size") or 0) < effect_threshold:
            continue
        qual.append(r)

    if not qual:
        print("ℹ️ No relations passed quality filters (effect/p-value/points).")
        # save empty outputs
        relations_json_path = os.path.join(out_dir, f"{dataset_name}_relations.json")
        sentences_json_path = os.path.join(out_dir, f"{dataset_name}_relations_sentences.json")
        with open(relations_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        with open(sentences_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return {"relations_path": relations_json_path, "sentences_path": sentences_json_path, "plots_dir": plots_dir}

    # Sort qualified relations by rank_score desc
    qual_sorted = sorted(qual, key=lambda x: x.get("rank_score", 0), reverse=True)

    # Determine plot caps based on targets and rules (strict)
    primary = targets_meta.get("primary")
    secondary = targets_meta.get("secondary") or []
    if isinstance(secondary, str):
        secondary = [secondary]
    # Rules:
    # only PRIMARY -> max 2
    # PRIMARY + 1 secondary -> max 4
    # PRIMARY + >=2 secondary -> max 5
    # no targets -> 1
    if primary and (not secondary or len(secondary) == 0):
        max_plots = 2
    elif primary and len(secondary) == 1:
        max_plots = 4
    elif primary and len(secondary) >= 2:
        max_plots = 5
    else:
        max_plots = 1
    # user also wanted "no more than 3 or 4" usually; clamp to 5
    max_plots = min(max_plots, 5)
    # enforce global max of 4 as "no more than 3 or 4" — user asked both; we'll honor stricter: 4
    max_plots = min(max_plots, 4)

    # but also don't create more plots than number of parents chooseable
    # ensure we do not plot duplicated parent involvement more than once per parent if possible:
    selected = []
    used_parents = set()
    for r in qual_sorted:
        pa, pb = r["parent_a"], r["parent_b"]
        # skip parent vs itself (just in case)
        if pa == pb:
            continue
        # skip if both parents already used and we prefer spread (prefer unique parents)
        if len(selected) < max_plots:
            # if either parent already used, we may still accept if necessary but prefer new parents
            if (pa in used_parents) and (pb in used_parents):
                # try skip to encourage variety
                continue
            selected.append(r)
            used_parents.add(pa)
            used_parents.add(pb)
        else:
            break

    # If selected is empty (rare), fallback to top relations up to max_plots
    if not selected:
        selected = qual_sorted[:max_plots]
    # if selected > max_plots trim
    selected = selected[:max_plots]

    # Now generate plots and prepare final results
    final_results = []
    for idx, r in enumerate(selected, start=1):
        # plot path
        plot_path = os.path.join(plots_dir, f"relation_{idx}.png")
        # plot using representative columns but label with prettified parent names
        plot_parent_relation(df, r, plot_path)
        # attach plot (relative)
        r["plot"] = os.path.relpath(plot_path, os.path.dirname(out_dir))
        # set final feature names to parent names (for JSON)
        #r["feature_a"] = r["parent_a"]
        #r["feature_b"] = r["parent_b"]
        final_results.append(r)

    # Sentences (use parent names)
    sentences = [relation_to_sentence_parent(r) for r in final_results]

    # Save results (only final selected relations are saved for plots + sentences)
    relations_json_path = os.path.join(out_dir, f"{dataset_name}_relations.json")
    sentences_json_path = os.path.join(out_dir, f"{dataset_name}_relations_sentences.json")
    summary_txt_path = os.path.join(out_dir, "summary.txt")

    with open(relations_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    with open(sentences_json_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("Top relations (final selected):\n")
        for i, r in enumerate(final_results, start=1):
            f.write(f"{i}. {prettify(r['feature_a'])} ↔ {prettify(r['feature_b'])} | method={r.get('method')} | effect={r.get('effect_size'):.3f} | p={r.get('p_value'):.3g} | adj_p={r.get('adjusted_p_value'):.3g}\n")
        f.write("\nNotes:\n")
        f.write("- Only high-quality relations (effect >= {0:.2f}, adj_p <= {1:.2f}, min_points = {2}) are plotted.\n".format(effect_threshold, pval_threshold, min_points))
        f.write("- Parent-level reporting: engineered columns are used for calculation but outputs reference original CSV column names.\n")
        f.write("- Plots labels use prettified parent names (underscores removed, Title Case).\n")

    print(f"✅ Relations saved → {relations_json_path}")
    print(f"✅ Sentences saved → {sentences_json_path}")
    print(f"✅ Plots saved → {plots_dir}")
    print(f"✅ Summary saved → {summary_txt_path}")

    return {"relations_path": relations_json_path, "sentences_path": sentences_json_path, "plots_dir": plots_dir}

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Relation & Visualization Analyzer (parent-level)")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--profile_path", required=True)
    parser.add_argument("--targets_meta_path", required=True)
    parser.add_argument("--sentiment_meta_path", required=True)
    parser.add_argument("--feature_meta_path", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    analyze_relations(
        dataset_name=args.dataset_name,
        features_path=args.features_path,
        profile_path=args.profile_path,
        targets_meta_path=args.targets_meta_path,
        sentiment_meta_path=args.sentiment_meta_path,
        feature_meta_path=args.feature_meta_path,
        out_dir=args.out_dir
    )
