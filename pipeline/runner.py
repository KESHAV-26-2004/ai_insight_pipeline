# pipeline/run_pipeline.py
import os, sys

sys.path.append(os.path.abspath("./pipeline"))
sys.path.append(os.path.abspath("./utils"))

from ingest_and_profile import profile_csv
from clean_and_validate import clean_and_validate
from detect_and_annotate_csv import detect_and_annotate_csv
from utils.io_utils import make_feature_config
from feature_engineer import main as feature_main
from pipeline.relation_analyzer import analyze_relations
from title_generator import generate_and_save_title
from gemini_refiner import generate_report_for_dataset
from final_report import generate_full_report


def run_pipeline(csv_path, primary_target, secondary_targets, task_type):
    """
    Runs the full pipeline end-to-end.
    Called by Streamlit UI.

    Args:
        csv_path: path to user-uploaded CSV file
        primary_target: string - selected target column
        secondary_targets: list[str] - up to 2 secondary targets
        task_type: "classification" or "regression"
    """

    base = os.path.basename(csv_path).replace(".csv", "")
    dataset_dir = f"./results/{base}"
    os.makedirs(dataset_dir, exist_ok=True)

    profile_path = f"{dataset_dir}/profiles/{base}_profile.json"
    cleaned_csv = f"{dataset_dir}/cleaned/{base}_cleaned.csv"
    enriched_csv = f"{dataset_dir}/enriched/{base}_cleaned_with_sentiment.csv"
    config_path = f"./configs/{base}_feature_config.json"
    title_path = f"{dataset_dir}/title/{base}_title.json"

    # STEP 1: Ingest & Profile
    print("\n===== STEP 1: Ingest & Profile =====")
    if not os.path.exists(profile_path):
        df, profile = profile_csv(csv_path, out_dir=f"{dataset_dir}/profiles")
    else:
        print(f"⏭️ Skipping profiling → {profile_path}")

    # STEP 1.5: Title Generation
    print("\n===== STEP 1.5: Generate Dataset Title =====")
    if not os.path.exists(title_path):
        generate_and_save_title(csv_path=csv_path, out_dir=f"{dataset_dir}/title")
    else:
        print(f"⏭️ Skipping title generation → {title_path}")

    # STEP 2: Clean & Validate
    print("\n===== STEP 2: Clean & Validate =====")
    if not os.path.exists(cleaned_csv):
        df_clean, targets_meta, profile = clean_and_validate(
            profile_path=profile_path,
            csv_path=csv_path,
            primary_target=primary_target,
            secondary_targets=secondary_targets,
            task_type=task_type,
            out_dir=f"{dataset_dir}/cleaned"
        )
    else:
        print(f"⏭️ Skipping cleaning → {cleaned_csv}")

    # STEP 3: Sentiment
    print("\n===== STEP 3: Detect & Annotate Sentiment =====")
    if not os.path.exists(enriched_csv):
        df_sent, meta = detect_and_annotate_csv(
            cleaned_csv,
            out_dir=f"{dataset_dir}/enriched",
            validate_existing_with_model=False
        )
    else:
        print(f"⏭️ Skipping sentiment → {enriched_csv}")

    # STEP 4: Feature Engineering
    print("\n===== STEP 4: Feature Engineering =====")
    if not os.path.exists(config_path):
        config_path = make_feature_config(base)

    feature_output_csv = f"{dataset_dir}/features/{base}_features.csv"
    if not os.path.exists(feature_output_csv):
        feature_main(config_path)
    else:
        print(f"⏭️ Skipping feature engineering → {feature_output_csv}")

    # STEP 5: Relations
    print("\n===== STEP 5: Relation Analysis =====")
    relations_json = f"{dataset_dir}/relations/{base}_relations.json"
    if not os.path.exists(relations_json):
        analyze_relations(
            dataset_name=base,
            features_path=feature_output_csv,
            profile_path=profile_path,
            targets_meta_path=f"{dataset_dir}/cleaned/{base}_cleaned_targets_meta.json",
            sentiment_meta_path=f"{dataset_dir}/enriched/{base}_sentiment_meta.json",
            feature_meta_path=f"{dataset_dir}/meta/{base}_encoders.json",
            out_dir=f"{dataset_dir}/relations",
        )
    else:
        print(f"⏭️ Skipping relation analysis → {relations_json}")

    # STEP 6: Gemini Summary Generation
    print("\n===== STEP 6: Gemini Summary Generation =====")
    gemini_output = f"{dataset_dir}/gemini/{base}_gemini_output.json"
    if not os.path.exists(gemini_output):
        try:
            generate_report_for_dataset(dataset_dir)
        except Exception as e:
            print("❌ Gemini failed:", e)
    else:
        print(f"⏭️ Skipping Gemini → {gemini_output}")

    # STEP 7: Final PDF/DOCX Report
    print("\n===== STEP 7: Final Report =====")
    try:
        out = generate_full_report(dataset_dir)
        print("📄 Final Report Generated:", out)
    except Exception as e:
        print("❌ Final report failed:", e)

    print("\n🎉 Pipeline complete:", base)
    return True
