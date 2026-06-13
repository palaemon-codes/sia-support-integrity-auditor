#!/usr/bin/env python3
"""
Support Integrity Auditor (SIA) — Inference Script
===================================================
MARS Open Projects 2026
Praneshwar Kannan Kommiya | 23117102 | B.Tech ME 4th Year

Accepts a CSV file of tickets and outputs:
  1. Binary mismatch predictions per ticket
  2. Full Evidence Dossiers for flagged tickets
  3. Summary statistics

Usage:
    python predict.py --input new_tickets.csv --output results/ --model models/

Requirements are listed in requirements.txt
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# CRITICAL: Import train_pipeline classes BEFORE joblib.load so pickle can find them
import train_pipeline
from train_pipeline import (
    MismatchClassifier, PseudoLabelGenerator, EvidenceDossierGenerator,
    PRIORITY_ORDER, PRIORITY_REVERSE,
)

warnings.filterwarnings("ignore")


def load_artifacts(model_dir: str) -> tuple:
    """Load trained model and pseudo-label generator from disk."""
    clf_path = os.path.join(model_dir, "mismatch_classifier_xgb.pkl")
    plg_path = os.path.join(model_dir, "pseudo_label_generator.pkl")

    if not os.path.exists(clf_path):
        print(f"[ERROR] Classifier not found at {clf_path}")
        print("Run train_pipeline.py first to train the model.")
        sys.exit(1)
    if not os.path.exists(plg_path):
        print(f"[ERROR] Pseudo-label generator not found at {plg_path}")
        print("Run train_pipeline.py first.")
        sys.exit(1)

    print(f"Loading classifier from {clf_path}...")
    artifacts = joblib.load(clf_path)
    
    # Reconstruct MismatchClassifier from components
    clf = MismatchClassifier(random_state=42)
    clf.model = artifacts["xgb_model"]
    clf.tfidf_vectorizer = artifacts["tfidf_vectorizer"]
    clf.label_encoder_channel = artifacts["label_encoder_channel"]
    clf.label_encoder_category = artifacts["label_encoder_category"]
    clf.scaler = artifacts["scaler"]
    clf.feature_names_ = artifacts.get("feature_names", [])
    clf._fitted = True

    print(f"Loading pseudo-label generator from {plg_path}...")
    plg = joblib.load(plg_path)

    return clf, plg


def predict_and_generate_dossiers(df: pd.DataFrame, clf, plg, output_dir: str):
    """Run full inference pipeline on input data."""
    from train_pipeline import EvidenceDossierGenerator, PRIORITY_ORDER, PRIORITY_REVERSE

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate pseudo-labels using pre-fitted models (refit=False)
    print("\n[Step 1] Generating pseudo-labels for input tickets (inference mode)...")
    df_labeled = plg.generate_labels(df, refit=False)

    # Step 2: Predict mismatches using trained classifier
    print("\n[Step 2] Running classifier predictions...")
    y_pred, y_proba = clf.predict(df_labeled)

    # Step 3: Generate evidence dossiers
    print("\n[Step 3] Generating evidence dossiers...")
    dg = EvidenceDossierGenerator(plg, clf)
    dossiers = dg.generate_batch(df_labeled, y_pred, y_proba)

    # Build output DataFrame with predictions
    df_output = df_labeled.copy()
    df_output["predicted_mismatch"] = y_pred
    df_output["mismatch_confidence"] = y_proba.round(4)

    # Save predictions CSV
    pred_path = os.path.join(output_dir, "predictions.csv")
    df_output.to_csv(pred_path, index=False)
    print(f"\n[Saved] Predictions → {pred_path}")

    # Save dossiers JSON
    dossiers_path = os.path.join(output_dir, "evidence_dossiers.json")
    with open(dossiers_path, "w") as f:
        json.dump(dossiers, f, indent=2)
    print(f"[Saved] Evidence dossiers → {dossiers_path} ({len(dossiers)} flagged)")

    # Save summary statistics
    summary = {
        "total_tickets": int(len(df_output)),
        "flagged_mismatches": int(y_pred.sum()),
        "flag_rate": float(round(y_pred.mean(), 4)),
        "hidden_crisis_count": int((df_output["mismatch_type"] == "Hidden Crisis").sum()),
        "false_alarm_count": int((df_output["mismatch_type"] == "False Alarm").sum()),
        "consistent_count": int((df_output["mismatch_type"] == "Consistent").sum()),
        "severity_distribution": df_output["inferred_severity"].value_counts().to_dict(),
        "channel_breakdown": df_output.groupby("Ticket_Channel")["predicted_mismatch"].mean().to_dict(),
        "category_breakdown": df_output.groupby("Issue_Category")["predicted_mismatch"].mean().to_dict(),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary → {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Total tickets processed:  {summary['total_tickets']}")
    print(f"Flagged as mismatches:    {summary['flagged_mismatches']} ({summary['flag_rate']*100:.1f}%)")
    print(f"  - Hidden Crisis:        {summary['hidden_crisis_count']}")
    print(f"  - False Alarm:          {summary['false_alarm_count']}")
    print(f"  - Consistent:           {summary['consistent_count']}")
    print(f"\nFlag rate by channel:")
    for ch, rate in summary["channel_breakdown"].items():
        print(f"  {ch}: {rate*100:.1f}%")
    print(f"\nFlag rate by category:")
    for cat, rate in summary["category_breakdown"].items():
        print(f"  {cat}: {rate*100:.1f}%")

    return df_output, dossiers


def main():
    parser = argparse.ArgumentParser(description="SIA Inference Script")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV with tickets to audit")
    parser.add_argument("--output", type=str, default="results/",
                        help="Output directory for predictions and dossiers")
    parser.add_argument("--model", type=str, default="models/",
                        help="Directory containing trained model artifacts")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Load data
    print(f"Loading input data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} tickets.")

    # Validate required columns
    required_cols = ["Ticket_ID", "Ticket_Subject", "Ticket_Description",
                     "Priority_Level", "Ticket_Channel", "Resolution_Time_Hours",
                     "Issue_Category", "Customer_Email"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing columns: {missing}")
        print("Some evidence signals may be incomplete.")

    # Load artifacts and run inference
    clf, plg = load_artifacts(args.model)
    predict_and_generate_dossiers(df, clf, plg, args.output)


if __name__ == "__main__":
    main()
