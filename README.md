# 🚀 **AI Insight Pipeline**

### *End-to-End Automated Data Analysis, Sentiment Detection, Feature Engineering, Insights & Reporting*

The **AI Insight Pipeline** is a fully automated system that transforms any CSV dataset into a complete analytical report.
It performs **cleaning, profiling, sentiment detection, feature engineering, correlation analysis, natural-language summaries, and generates professional PDF/DOCX reports** — all in one run.

This project takes a dataset and guides the user through:

1. Selecting **Primary Target**
2. Selecting **Secondary Target(s)** (optional, up to 2)
3. Choosing **Task Type** (Classification or Regression)
4. Running a full automated pipeline

> **Goal:** Make dataset analysis automatic, intelligent, and production-ready — no manual coding required.

---

# 📌 **✨ Features Overview**

### 🔹 1. **CSV Profiling**

* Reads dataset safely with encoding detection
* Creates full JSON profile: data types, sample values, unique counts, missing values
* Detects numeric, categorical, text, and datetime columns

---

### 🔹 2. **Dataset Title Generation (Fine-Tuned GPT-2 Model)**

* Summarizes CSV structure
* Detects dataset domain
* Generates a professional dataset title using a fine-tuned GPT-2 model
* Saves title as JSON under `results/<dataset>/title/`

---

### 🔹 3. **Clean & Validate**

User provides:

* **Primary Target**
* **Secondary Targets** (0–2 optional)
* **Task Type**: `classification` or `regression`

Pipeline then:

* Cleans invalid/unwanted columns
* Normalizes datetime formats
* Fixes inconsistent data types
* Automatically collapses giant categorical classes
* Saves cleaned CSV + target metadata

---

### 🔹 4. **Sentiment Detection**

The pipeline detects sentiment columns in two ways:

#### ✔ Case A — **Existing Sentiment Column Found**

If dataset contains columns like `positive/negative/neutral`, pipeline:

* Converts them to numeric (`0=neg`, `1=neu`, `2=pos`)
* Saves metadata
* Adds them as useful secondary features

#### ✔ Case B — **No Sentiment → Use Custom Fine-Tuned BERT Models**

Runs two in-house models:

* **Opinion Detector** (detects if text contains opinion)
* **DistilBERT Sentiment Model** (positive/neutral/negative)

✓ Auto-batching
✓ GPU acceleration
✓ Smart text cleaning
✓ Caching for speed

Adds:

* `<column>_sentiment`
* `<column>_sentiment_num`
* `<column>_sentiment_confidence`

---

### 🔹 5. **Feature Engineering**

Automatically applies:

* Numeric scaling (z-score)
* Label encoding / frequency encoding
* Text metrics (word count, unique ratio, emoji, URLs, uppercase ratio)
* Datetime expansion (year/month/day/hour/weekday)
* Aggregated sentiment features
* Row-level meta features

Outputs:

* `<dataset>_features.csv`
* `<dataset>_encoders.json` metadata

---

### 🔹 6. **Relation Analysis (Parent-Level Correlations)**

The Relation Analyzer:

* Maps engineered features back to original “parent” columns
* Picks the **best representative feature** per parent column
* Computes:

  * Pearson/Spearman
  * ANOVA Eta²
  * Chi-square / Cramer’s V
* Applies strict quality filters:

  * effect size ≥ 0.15
  * corrected p-value ≤ 0.05
  * sample size thresholds
* Selects **3–4 best relations**
* Generates **visual plots**
* Produces natural-language explanation sentences

Outputs:

* `relations.json`
* `relations_sentences.json`
* Plot images in `results/<dataset>/plots/`

---

### 🔹 7. **Gemini AI Summary Generation**

Using Gemini 2.5 Flash:

* Builds full dataset prompt
* Generates:

  * Final summary (6–10 sentences)
  * 5–12 actionable recommendations
* Cleans output
* Saves:

  * JSON
  * Markdown report

---

### 🔹 8. **Final Report Generator → PDF + DOCX**

Creates three polished formats:

* **Markdown**
* **DOCX**
* **PDF (with clean layout)**

Report includes:

1. Dataset Title
2. Overview
3. Column details with samples
4. Sentiment distribution (pie chart)
5. Key drivers & correlations (with plots)
6. Final AI summary
7. Recommendations

---

# 📂 **Folder Structure**

```
root/
│
├── pipeline/
│   ├── title_generator.py
│   ├── run_pipeline.py
│   ├── relation_analyzer.py
│   ├── ingest_and_profile.py
│   ├── detect_and_annotate_csv.py
│   ├── feature_engineer.py
│   ├── final_report.py
│   └── gemini_refiner.py
│
├── Sentiment/
│   ├── opinion_detector_model/ (ignored)
│   ├── sentiment_model/ (ignored)
│   └── datasets & training notebooks
│
│
├── title/
│   ├── model/ (fine-tuned GPT-2) (ignored)
│   ├── data/ (ignored)
│   └── training notebook
│
├── data/all_csv
│   └── (user CSV files – ignored)
│
├── results/
│   └── <dataset>/
│       ├── profiles/
│       ├── cleaned/
│       ├── enriched/
│       ├── features/
│       ├── relations/
│       ├── gemini/
│       └── report/
│
├── app.py (optional UI)
├── app.ipynb
├── README.md
└── requirements.txt
```

---

# 🧪 **How to Run the Pipeline**

### Step 1 — Place your CSV inside:

```
data/all_csv
```

### Step 2 — Run the pipeline:

```bash
app.ipynb
```

### The program will ask you:

---

## **1. Primary Target (required)**

Choose the main outcome column.
Example:

```
Enter primary target: rating
```

---

## **2. Secondary Targets (optional, max 2)**

```
Enter secondary targets (comma separated, blank for none):
sentiment, product_category
```

---

## **3. Task Type**

Classification or regression:

```
Enter task type (classification/regression): classification
```

---

### 🚀 After this — everything else is automatic.

Pipeline output will be created inside:

```
results/<dataset>/
```

Including:

* Title JSON
* Cleaned dataset
* Enriched sentiment CSV
* Feature-engineered CSV
* Correlation plots
* AI summary
* Final **PDF + DOCX + MD reports**

---

# 📦 **Installation**

```bash
pip install -r requirements.txt
```

Make sure to set:

```
export GOOGLE_API_KEY="your key"
```

---

# ❤️ **Contributions**

Feel free to submit PRs for:

* new sentiment models
* additional relation metrics
* UI improvements
* extended report templates

---

# 🏁 **License**

MIT License
