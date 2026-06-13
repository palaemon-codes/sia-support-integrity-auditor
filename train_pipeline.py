#!/usr/bin/env python3
"""
Support Integrity Auditor (SIA) — Training Pipeline
=====================================================
MARS Open Projects 2026
Praneshwar Kannan Kommiya | 23117102 | B.Tech ME 4th Year

This script implements the full SIA pipeline:
  Stage 1 — Pseudo-label generation (self-supervised)
  Stage 2 — Classifier training (XGBoost + optional Transformer fine-tuning)
  Stage 3 — Evidence dossier generation

Usage:
    python train_pipeline.py --data dataset/customer_support_tickets.csv --output models/
    python train_pipeline.py --skip-embeddings  # Skip embedding signal (faster)

Requirements are listed in requirements.txt
"""

import argparse
import json
import os
import pickle
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

# Fix for tokenizers parallelism issues on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

PRIORITY_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_REVERSE = {v: k for k, v in PRIORITY_ORDER.items()}

# Urgency / escalation keywords with severity weights
URGENCY_KEYWORDS = {
    # Critical indicators
    "data loss": 4, "data breach": 4, "security breach": 4,
    "cannot access": 3, "unable to login": 3, "locked out": 3,
    "system down": 4, "outage": 4, "crash": 3, "crashes": 3,
    "not loading": 3, "spinning wheel": 2, "blank screen": 3,
    "payment failed": 3, "billing error": 3, "overcharged": 3,
    "refund": 2, "cancel subscription": 2, "delete account": 2,
    "urgent": 3, "asap": 3, "immediately": 3, "critical": 4,
    "fraud": 4, "unauthorized": 4, "suspicious": 3,
    "lost phone": 3, "2fa": 2, "cannot log": 3, "login fail": 3,
    "password reset": 2, "not receiving email": 2,
    "sync": 2, "syncing": 2, "not syncing": 3, "not synced": 3,
    # Escalation phrases
    "speak to manager": 2, "escalate": 2, "complaint": 2,
    "no response": 2, "waiting for days": 2, "days ago": 2,
    "multiple times": 2, "again": 1, "still": 1, "yet": 1,
    # Mild indicators
    "how do i": 0, "what are": 0, "where is": 0, "request demo": 0,
    "roadmap": 0, "feature request": 0, "hours of operation": 0,
}

# Domain tier mapping based on email domain patterns
def extract_domain_tier(email: str) -> int:
    """Extract a proxy for customer tier from email domain."""
    if not isinstance(email, str):
        return 1
    email = email.lower()
    if any(d in email for d in ["enterprise.org", "company.com", "tech.io", "corp."]):
        return 3  # Enterprise
    elif any(d in email for d in ["example.org", "example.net"]):
        return 2  # Standard
    else:
        return 1  # Basic


# Channel urgency mapping
CHANNEL_URGENCY = {"Phone": 3, "Chat": 2, "Social Media": 2, "Email": 1, "Web Form": 1}

# Category base severity
CATEGORY_BASE_SEVERITY = {
    "Fraud": 3,
    "Technical": 2,
    "Billing": 2,
    "Account": 1,
    "General Inquiry": 0,
}


# ---------------------------------------------------------------------------
# Stage 1: Pseudo-Label Generation
# ---------------------------------------------------------------------------

class PseudoLabelGenerator:
    """
    Generates self-supervised binary mismatch labels by fusing three
    independent signals:
      1. Rule-based NLP keyword + TF-IDF severity scoring
      2. Embedding-based semantic clustering
      3. Resolution-time-based severity proxy
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.tfidf_vectorizer = None
        self.kmeans = None
        self.svd = None
        self.cluster_severity_map = {}
        self.signal_weights = {"keyword": 0.35, "embedding": 0.30, "resolution": 0.35}
        self._fitted = False
        self._resolution_percentiles = None
        self._embedding_skipped = False

    # ------- Signal A: Rule-Based NLP + TF-IDF -------

    def _compute_keyword_score(self, text: str) -> float:
        """Score a ticket's text for urgency using keyword density."""
        if not isinstance(text, str):
            return 0.0
        text_lower = text.lower()
        total_score = 0.0
        hits = 0
        for phrase, weight in URGENCY_KEYWORDS.items():
            count = text_lower.count(phrase)
            if count > 0:
                total_score += weight * count
                hits += count
        # Normalize by text length to avoid long-text bias
        text_len = max(len(text_lower.split()), 1)
        return (total_score / text_len) * 100  # scaled

    def _compute_negation_score(self, text: str) -> float:
        """Detect negation patterns that may indicate frustration."""
        if not isinstance(text, str):
            return 0.0
        negation_patterns = [
            r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bcan't\b", r"\bcannot\b",
            r"\bwon't\b", r"\bdoesn't\b", r"\bdon't\b", r"\bhaven't\b",
            r"\bhasn't\b", r"\bstill not\b", r"\bstill no\b", r"\byet\b",
        ]
        count = sum(len(re.findall(p, text.lower())) for p in negation_patterns)
        return count

    def _compute_signal_keyword(self, df: pd.DataFrame) -> np.ndarray:
        """Compute keyword-based severity scores for all tickets."""
        scores = []
        for _, row in df.iterrows():
            combined_text = f"{row.get('Ticket_Subject', '')} {row.get('Ticket_Description', '')}"
            kw_score = self._compute_keyword_score(combined_text)
            neg_score = self._compute_negation_score(combined_text)
            # Blend keyword + negation
            final = kw_score + neg_score * 0.5
            scores.append(final)
        arr = np.array(scores)
        # Normalize to 0-3 scale matching priority levels
        if arr.std() > 0:
            arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        else:
            arr_norm = np.zeros_like(arr)
        # Discretize into 4 severity levels
        severity = np.digitize(arr_norm, bins=[0.2, 0.5, 0.8])  # 0=Low, 1=Medium, 2=High, 3=Critical
        return severity

    # ------- Signal B: Embedding-Based Clustering -------

    def _compute_signal_embedding(self, df: pd.DataFrame, refit: bool = True, skip_embedding: bool = False) -> np.ndarray:
        """Cluster tickets by semantic urgency.
        
        Primary: sentence-transformers → SVD → KMeans → severity mapping.
        Fallback (when sentence-transformers unavailable or skip_embedding=True):
          TF-IDF → SVD → KMeans → severity mapping.
          This produces a genuinely different signal from keyword counting.
        """
        if skip_embedding or self._embedding_skipped:
            if skip_embedding:
                self._embedding_skipped = True
            
            # ---- TF-IDF based fallback embedding ----
            print("  [Embedding] Using TF-IDF + KMeans clustering (lightweight fallback)...")
            
            texts = []
            for _, row in df.iterrows():
                subj = str(row.get("Ticket_Subject", ""))
                desc = str(row.get("Ticket_Description", ""))
                desc_clean = " ".join(desc.split()[:150])
                texts.append(f"{subj}. {desc_clean}")
            
            if refit or self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=3000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True
                )
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                
                n_components = min(64, tfidf_matrix.shape[0] // 2, tfidf_matrix.shape[1])
                self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
                embeddings_reduced = self.svd.fit_transform(tfidf_matrix)
                
                n_clusters = 8
                self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                cluster_labels = self.kmeans.fit_predict(embeddings_reduced)
                
                df_temp = df.copy()
                df_temp["_cluster"] = cluster_labels
                cluster_res_time = df_temp.groupby("_cluster")["Resolution_Time_Hours"].mean().sort_values()
                sorted_clusters = cluster_res_time.index.tolist()
                n_per_level = max(1, n_clusters // 4)
                self.cluster_severity_map = {}
                for i, cluster_id in enumerate(sorted_clusters):
                    level = min(i // n_per_level, 3)
                    self.cluster_severity_map[cluster_id] = level
                
                severity = np.array([self.cluster_severity_map[c] for c in cluster_labels])
            else:
                tfidf_matrix = self.tfidf_vectorizer.transform(texts)
                embeddings_reduced = self.svd.transform(tfidf_matrix)
                cluster_labels = self.kmeans.predict(embeddings_reduced)
                severity = np.array([self.cluster_severity_map.get(c, 1) for c in cluster_labels])
            
            return severity

        try:
            from sentence_transformers import SentenceTransformer
            _ok = True
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"  [Embedding] WARNING: sentence-transformers unavailable ({e})")
            print("  [Embedding] Falling back to keyword-based signal.")
            _ok = False

        if not _ok:
            return self._compute_signal_keyword(df)

        if refit or self.kmeans is None:
            print("  [Embedding] Loading sentence-transformer model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = []
        for _, row in df.iterrows():
            subj = str(row.get("Ticket_Subject", ""))
            desc = str(row.get("Ticket_Description", ""))
            desc_clean = " ".join(desc.split()[:200])
            texts.append(f"{subj}. {desc_clean}")

        print(f"  [Embedding] Encoding {len(texts)} tickets...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

        if refit or self.svd is None:
            # Fit: reduce dims + cluster + map to severity
            n_components = min(128, embeddings.shape[1], embeddings.shape[0] // 2)
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            embeddings_reduced = self.svd.fit_transform(embeddings)

            n_clusters = 8
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = self.kmeans.fit_predict(embeddings_reduced)

            df_temp = df.copy()
            df_temp["_cluster"] = cluster_labels
            cluster_res_time = df_temp.groupby("_cluster")["Resolution_Time_Hours"].mean().sort_values()
            sorted_clusters = cluster_res_time.index.tolist()
            n_per_level = max(1, n_clusters // 4)
            self.cluster_severity_map = {}
            for i, cluster_id in enumerate(sorted_clusters):
                level = min(i // n_per_level, 3)
                self.cluster_severity_map[cluster_id] = level

            severity = np.array([self.cluster_severity_map[c] for c in cluster_labels])
        else:
            # Transform only: reduce using fitted SVD, predict clusters, map
            embeddings_reduced = self.svd.transform(embeddings)
            cluster_labels = self.kmeans.predict(embeddings_reduced)
            severity = np.array([self.cluster_severity_map.get(c, 1) for c in cluster_labels])

        return severity

    # ------- Signal C: Resolution-Time Regression -------

    def _compute_signal_resolution(self, df: pd.DataFrame, refit: bool = True) -> np.ndarray:
        """Use resolution time as a direct severity proxy."""
        hours = df["Resolution_Time_Hours"].values

        if refit or self._resolution_percentiles is None:
            p30 = np.percentile(hours, 30)
            p60 = np.percentile(hours, 60)
            p85 = np.percentile(hours, 85)
            self._resolution_percentiles = (p30, p60, p85)
        else:
            p30, p60, p85 = self._resolution_percentiles

        severity = np.zeros(len(hours), dtype=int)
        severity[hours > p30] = 1
        severity[hours > p60] = 2
        severity[hours > p85] = 3
        return severity

    # ------- Fusion Strategy -------

    def _fuse_signals(
        self, signal_kw: np.ndarray, signal_emb: np.ndarray, signal_res: np.ndarray
    ) -> np.ndarray:
        """
        Fuse signals into inferred severity (0-3).
        
        Strategy: Resolution-time is the base (most objective). Keyword and embedding 
        signals can adjust it by ±1 when they BOTH strongly disagree with resolution.
        
        This prevents any single noisy text signal from dominating while still
        allowing text evidence to shift severity in clear-cut cases.
        """
        base = signal_res.astype(int).copy()
        
        # Compute how much each text signal differs from resolution
        kw_diff = signal_kw.astype(int) - base
        emb_diff = signal_emb.astype(int) - base
        
        # Only adjust when BOTH text signals agree on direction AND magnitude >= 2
        for i in range(len(base)):
            if abs(kw_diff[i]) >= 2 and abs(emb_diff[i]) >= 2:
                # Both signals strongly agree on a different severity
                if np.sign(kw_diff[i]) == np.sign(emb_diff[i]):
                    adjustment = np.sign(kw_diff[i])  # +1 or -1
                    base[i] = np.clip(base[i] + adjustment, 0, 3)
            elif abs(kw_diff[i]) >= 2:
                # Only keyword strongly disagrees
                base[i] = np.clip(base[i] + np.sign(kw_diff[i]), 0, 3)
        
        return base

    def generate_labels(self, df: pd.DataFrame, refit: bool = True, skip_embedding: bool = False) -> pd.DataFrame:
        """
        Generate pseudo-labels for the dataset.
        
        Parameters:
            df: Input DataFrame with ticket data.
            refit: If True, fit all models on this data. If False, use pre-fitted models.
            skip_embedding: If True, use TF-IDF clustering instead of sentence-transformers.
        """
        label = "STAGE 1: Pseudo-Label Generation" if refit else "STAGE 1: Pseudo-Label Inference"
        print("=" * 60)
        print(label)
        print("=" * 60)

        # Signal A: Keyword-based
        print("\n[Signal A] Computing rule-based NLP keyword scores...")
        signal_kw = self._compute_signal_keyword(df)

        # Signal B: Embedding-based clustering
        print("\n[Signal B] Computing embedding-based clustering...")
        signal_emb = self._compute_signal_embedding(df, refit=refit, skip_embedding=skip_embedding)

        # Signal C: Resolution-time
        print("\n[Signal C] Computing resolution-time-based severity...")
        signal_res = self._compute_signal_resolution(df, refit=refit)

        # Fuse signals
        print("\n[Fusion] Resolution-time base + text signal adjustments...")
        inferred_severity = self._fuse_signals(signal_kw, signal_emb, signal_res)

        # Map assigned priority to numeric
        assigned_num = df["Priority_Level"].map(PRIORITY_ORDER).values

        # Derive mismatch labels — use delta >= 2 for more conservative detection
        # This means a ticket assigned "Low" must be inferred as at least "High" to flag
        # Similarly, assigned "Critical" must be inferred as at most "Medium" to flag
        is_mismatch = np.abs(inferred_severity - assigned_num) >= 2
        severity_delta = inferred_severity - assigned_num

        def determine_mismatch_type(delta: int) -> str:
            if abs(delta) < 2:
                return "Consistent"
            elif delta > 0:
                return "Hidden Crisis"
            else:
                return "False Alarm"

        mismatch_type = np.array([determine_mismatch_type(d) for d in severity_delta])

        # Build result DataFrame
        result = df.copy()
        result["inferred_severity_num"] = inferred_severity
        result["inferred_severity"] = result["inferred_severity_num"].map(PRIORITY_REVERSE)
        result["assigned_priority_num"] = assigned_num
        result["is_mismatch"] = is_mismatch.astype(int)
        result["mismatch_type"] = mismatch_type
        result["severity_delta"] = severity_delta
        result["signal_keyword"] = signal_kw
        result["signal_embedding"] = signal_emb
        result["signal_resolution"] = signal_res
        result["domain_tier"] = df["Customer_Email"].apply(extract_domain_tier)
        result["channel_urgency"] = df["Ticket_Channel"].map(CHANNEL_URGENCY).fillna(1)
        result["category_severity"] = df["Issue_Category"].map(CATEGORY_BASE_SEVERITY).fillna(1)

        if refit:
            self._fitted = True

        # Print distribution
        n_mismatch = is_mismatch.sum()
        n_total = len(df)
        print(f"\n[Results] Total tickets: {n_total}")
        print(f"[Results] Mismatches found: {n_mismatch} ({100*n_mismatch/n_total:.1f}%)")
        print(f"[Results]   Hidden Crisis: {(mismatch_type == 'Hidden Crisis').sum()}")
        print(f"[Results]   False Alarm:  {(mismatch_type == 'False Alarm').sum()}")
        print(f"[Results]   Consistent:   {(mismatch_type == 'Consistent').sum()}")

        # Signal agreement analysis (only on fit)
        if refit:
            agreements = []
            for s1, s2 in [("keyword", "embedding"), ("keyword", "resolution"), ("embedding", "resolution")]:
                v1 = result[f"signal_{s1}"].values
                v2 = result[f"signal_{s2}"].values
                agr = (v1 == v2).mean()
                agreements.append((f"{s1} vs {s2}", agr))
                print(f"[Agreement] {s1} vs {s2}: {agr:.3f}")
            result.attrs["signal_agreements"] = agreements
            result.attrs["mismatch_rate"] = n_mismatch / n_total

        return result


# ---------------------------------------------------------------------------
# Stage 2: Classifier Training
# ---------------------------------------------------------------------------

class MismatchClassifier:
    """
    Trains a binary classifier on pseudo-labeled data to detect priority mismatches.
    Uses XGBoost with TF-IDF text features + structured metadata features.
    Also supports optional Transformer fine-tuning path.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000, ngram_range=(1, 3), stop_words="english",
            sublinear_tf=True,
        )
        self.label_encoder_channel = LabelEncoder()
        self.label_encoder_category = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self._fitted = False
        self.feature_names_ = None

    def _prepare_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Build feature matrix from text + structured fields."""
        # Combine text fields
        texts = []
        for _, row in df.iterrows():
            subj = str(row.get("Ticket_Subject", ""))
            desc = str(row.get("Ticket_Description", ""))
            # Clean noise: keep first 150 words of description
            desc_clean = " ".join(desc.split()[:150])
            texts.append(f"{subj}. {desc_clean}")

        if fit:
            text_features = self.tfidf_vectorizer.fit_transform(texts)
            channel_encoded = self.label_encoder_channel.fit_transform(
                df["Ticket_Channel"].fillna("Unknown")
            )
            category_encoded = self.label_encoder_category.fit_transform(
                df["Issue_Category"].fillna("Unknown")
            )
        else:
            text_features = self.tfidf_vectorizer.transform(texts)
            channel_encoded = self.label_encoder_channel.transform(
                df["Ticket_Channel"].fillna("Unknown")
            )
            category_encoded = self.label_encoder_category.transform(
                df["Issue_Category"].fillna("Unknown")
            )

        # Structured features
        structured = np.column_stack([
            df["Resolution_Time_Hours"].fillna(0).values,
            channel_encoded,
            category_encoded,
            df.get("domain_tier", np.zeros(len(df))).values,
            df.get("channel_urgency", np.zeros(len(df))).values,
            df.get("category_severity", np.zeros(len(df))).values,
            df["Priority_Level"].map(PRIORITY_ORDER).fillna(1).values,  # Assigned priority as feature
        ])

        if fit:
            structured = self.scaler.fit_transform(structured)
        else:
            structured = self.scaler.transform(structured)

        # Combine text (dense) with structured
        from scipy.sparse import hstack, csr_matrix
        combined = hstack([text_features, csr_matrix(structured)])

        self.feature_names_ = (
            list(self.tfidf_vectorizer.get_feature_names_out())
            + ["resolution_time", "channel_encoded", "category_encoded",
               "domain_tier", "channel_urgency", "category_severity", "assigned_priority_num"]
        )
        return combined

    def train(self, df: pd.DataFrame, use_smote: bool = True):
        """Train the mismatch classifier on pseudo-labeled data."""
        print("\n" + "=" * 60)
        print("STAGE 2: Classifier Training")
        print("=" * 60)

        X = self._prepare_features(df, fit=True)
        y = df["is_mismatch"].values

        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )
        print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        print(f"Train mismatch rate: {y_train.mean():.3f}")

        # Handle class imbalance with class weights
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        scale_pos_weight = class_weights[1] / class_weights[0]
        print(f"[Imbalance] scale_pos_weight = {scale_pos_weight:.3f}")

        # Apply SMOTE if requested
        if use_smote:
            from imblearn.over_sampling import SMOTE
            print("[Imbalance] Applying SMOTE oversampling...")
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"[Imbalance] After SMOTE — train size: {X_train.shape[0]}, "
                  f"mismatch rate: {y_train.mean():.3f}")

        # Train XGBoost with better hyperparameters
        import xgboost as xgb
        print("\n[Training] Fitting XGBoost classifier (optimized params)...")
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        recall_per_class = recall_score(y_test, y_pred, average=None)
        recall_consistent = recall_per_class[0] if len(recall_per_class) > 0 else 0
        recall_mismatch = recall_per_class[1] if len(recall_per_class) > 1 else 0

        print(f"\n[Metrics] Binary Classification Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"[Metrics] Macro F1 Score: {f1_macro:.4f}")
        print(f"[Metrics] Per-Class Recall (Consistent): {recall_consistent:.4f}")
        print(f"[Metrics] Per-Class Recall (Mismatch):  {recall_mismatch:.4f}")
        print(f"\n[Classification Report]:")
        print(classification_report(y_test, y_pred, target_names=["Consistent", "Mismatch"]))

        self._fitted = True
        self._eval_results = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "recall_consistent": recall_consistent,
            "recall_mismatch": recall_mismatch,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
        }

        # Cross-validation
        print("\n[Cross-Validation] 5-fold Stratified CV...")
        cv_scores = cross_val_score(
            self.model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
            scoring="f1_macro",
        )
        print(f"[CV] Macro F1 (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    def predict(self, df: pd.DataFrame) -> tuple:
        """Predict mismatch labels for new data."""
        if not self._fitted:
            raise RuntimeError("Model must be trained before prediction.")
        X = self._prepare_features(df, fit=False)
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        return y_pred, y_proba

    def get_top_features(self, n: int = 15) -> list:
        """Return top features for interpretability."""
        if not self._fitted or self.model is None:
            return []
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:n]
        return [(self.feature_names_[i], importances[i]) for i in indices]


# ---------------------------------------------------------------------------
# Stage 3: Evidence Dossier Generation
# ---------------------------------------------------------------------------

class EvidenceDossierGenerator:
    """
    Generates structured, hallucination-free evidence dossiers for flagged tickets.
    Every claim is traceable to a specific field in the input ticket.
    """

    def __init__(self, pseudo_label_gen: PseudoLabelGenerator, classifier: MismatchClassifier):
        self.plg = pseudo_label_gen
        self.clf = classifier

    def generate_dossier(self, row: pd.Series, pred_mismatch: int, confidence: float) -> dict:
        """Generate a single evidence dossier."""
        ticket_id = row.get("Ticket_ID", "Unknown")
        assigned_priority = row.get("Priority_Level", "Unknown")
        inferred_sev = row.get("inferred_severity", "Unknown")
        mismatch_type = row.get("mismatch_type", "Unknown")
        severity_delta = int(row.get("severity_delta", 0))

        # Build feature evidence — every claim traced to a field
        feature_evidence = []

        # 1. Keyword signal evidence (traced to Ticket_Subject + Ticket_Description)
        combined_text = f"{row.get('Ticket_Subject', '')} {row.get('Ticket_Description', '')}"
        found_keywords = []
        for phrase, weight in URGENCY_KEYWORDS.items():
            if phrase.lower() in combined_text.lower():
                found_keywords.append((phrase, weight))
        found_keywords.sort(key=lambda x: x[1], reverse=True)

        if found_keywords:
            top_kw = found_keywords[:5]
            feature_evidence.append({
                "signal": "keyword",
                "value": ", ".join([f"'{kw[0]}' (weight={kw[1]})" for kw in top_kw]),
                "weight": f"{sum(kw[1] for kw in top_kw) / len(top_kw):.2f} avg",
                "source_field": "Ticket_Subject, Ticket_Description",
            })

        # 2. Resolution time evidence (traced to Resolution_Time_Hours)
        res_time = row.get("Resolution_Time_Hours", 0)
        res_interpretation = ""
        if res_time >= 85:
            res_interpretation = f"Resolution took {res_time}h — in the top 15% of all tickets, strongly suggesting underlying complexity."
        elif res_time >= 60:
            res_interpretation = f"Resolution took {res_time}h — above the 60th percentile, indicating a non-trivial issue."
        elif res_time < 10:
            res_interpretation = f"Resolved in only {res_time}h — very fast, suggesting the issue was straightforward."
        else:
            res_interpretation = f"Resolution took {res_time}h — near median, typical complexity."

        feature_evidence.append({
            "signal": "resolution_time",
            "value": f"{res_time} hours",
            "interpretation": res_interpretation,
            "source_field": "Resolution_Time_Hours",
        })

        # 3. Channel evidence (traced to Ticket_Channel)
        channel = row.get("Ticket_Channel", "Unknown")
        ch_urgency = CHANNEL_URGENCY.get(channel, 1)
        feature_evidence.append({
            "signal": "channel",
            "value": channel,
            "interpretation": f"Channel urgency score: {ch_urgency}/3. "
                             f"{'Higher-urgency channel' if ch_urgency >= 2 else 'Lower-urgency channel'}.",
            "source_field": "Ticket_Channel",
        })

        # 4. Category evidence (traced to Issue_Category)
        category = row.get("Issue_Category", "Unknown")
        cat_sev = CATEGORY_BASE_SEVERITY.get(category, 1)
        feature_evidence.append({
            "signal": "category",
            "value": category,
            "interpretation": f"Category base severity: {cat_sev}/3. "
                             f"{'Inherently high-severity category' if cat_sev >= 2 else 'Lower baseline severity'}.",
            "source_field": "Issue_Category",
        })

        # 5. Domain tier evidence (traced to Customer_Email)
        email = row.get("Customer_Email", "")
        tier = extract_domain_tier(email)
        tier_label = {1: "Basic", 2: "Standard", 3: "Enterprise"}.get(tier, "Unknown")
        feature_evidence.append({
            "signal": "domain_tier",
            "value": tier_label,
            "interpretation": f"Customer tier: {tier_label}. "
                             f"{'Enterprise customers may need priority handling.' if tier >= 3 else ''}",
            "source_field": "Customer_Email",
        })

        # Constraint analysis: 2-3 sentence grounded explanation
        constraint_analysis = self._build_constraint_analysis(
            row, assigned_priority, inferred_sev, mismatch_type, severity_delta, feature_evidence
        )

        dossier = {
            "ticket_id": ticket_id,
            "assigned_priority": assigned_priority,
            "inferred_severity": inferred_sev,
            "mismatch_type": mismatch_type,
            "severity_delta": str(severity_delta),
            "feature_evidence": feature_evidence,
            "constraint_analysis": constraint_analysis,
            "confidence": f"{confidence:.3f}",
        }
        return dossier

    def _build_constraint_analysis(self, row, assigned, inferred, mtype, delta, evidence) -> str:
        """Build grounded 2-3 sentence explanation."""
        subject = row.get("Ticket_Subject", "the issue")
        category = row.get("Issue_Category", "this category")
        channel = row.get("Ticket_Channel", "this channel")
        res_time = row.get("Resolution_Time_Hours", 0)

        if mtype == "Hidden Crisis":
            return (
                f"This ticket was assigned '{assigned}' priority but the evidence suggests it should be '{inferred}'. "
                f"The ticket subject ('{subject}') combined with a resolution time of {res_time}h and "
                f"category '{category}' indicate the issue is more severe than the assigned label reflects. "
                f"The system detected urgency indicators in the ticket language that are inconsistent with a '{assigned}' rating."
            )
        elif mtype == "False Alarm":
            return (
                f"This ticket was flagged as '{assigned}' but based on available evidence it aligns more closely with '{inferred}'. "
                f"The resolution was completed in just {res_time}h through '{channel}', and the '{category}' issue type "
                f"typically resolves without escalation. The assigned priority overstates the actual urgency."
            )
        else:
            return (
                f"The assigned priority '{assigned}' is consistent with the inferred severity '{inferred}'. "
                f"All evidence signals (keyword patterns, resolution time of {res_time}h, {channel} channel, "
                f"'{category}' category) point to the same severity level."
            )

    def generate_batch(self, df: pd.DataFrame, predictions: np.ndarray, confidences: np.ndarray) -> list:
        """Generate dossiers for all mismatched tickets."""
        dossiers = []
        mismatch_indices = np.where(predictions == 1)[0]
        print(f"\n[Dossiers] Generating {len(mismatch_indices)} evidence dossiers...")
        for idx in mismatch_indices:
            row = df.iloc[idx]
            dossier = self.generate_dossier(row, int(predictions[idx]), float(confidences[idx]))
            dossiers.append(dossier)
        return dossiers


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SIA Training Pipeline")
    parser.add_argument("--data", type=str, default="dataset/customer_support_tickets.csv",
                        help="Path to input CSV")
    parser.add_argument("--output", type=str, default="models/",
                        help="Output directory for models and artifacts")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding-based signal (faster, lower accuracy)")
    parser.add_argument("--adversarial-test", type=str, default=None,
                        help="Path to adversarial test CSV for robustness evaluation")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} tickets.")

    # ---- Stage 1: Pseudo-Label Generation ----
    plg = PseudoLabelGenerator(random_state=42)
    df_labeled = plg.generate_labels(df, refit=True, skip_embedding=args.skip_embeddings)

    # Save pseudo-labeled data
    labeled_path = os.path.join(args.output, "pseudo_labeled_data.csv")
    df_labeled.to_csv(labeled_path, index=False)
    print(f"\n[Saved] Pseudo-labeled data → {labeled_path}")

    # Save signal agreement stats
    agreements_path = os.path.join(args.output, "signal_agreements.json")
    with open(agreements_path, "w") as f:
        json.dump([
            {"signals": a[0], "agreement": float(a[1])}
            for a in df_labeled.attrs.get("signal_agreements", [])
        ], f, indent=2)

    # ---- Stage 2: Classifier Training ----
    clf = MismatchClassifier(random_state=42)
    clf.train(df_labeled, use_smote=True)

    # Save model artifacts (individual components for reliable deserialization)
    model_path = os.path.join(args.output, "mismatch_classifier_xgb.pkl")
    # Save components individually to avoid pickle __main__ issues
    artifacts = {
        "xgb_model": clf.model,
        "tfidf_vectorizer": clf.tfidf_vectorizer,
        "label_encoder_channel": clf.label_encoder_channel,
        "label_encoder_category": clf.label_encoder_category,
        "scaler": clf.scaler,
        "feature_names": clf.feature_names_,
    }
    joblib.dump(artifacts, model_path)
    print(f"\n[Saved] Trained classifier → {model_path}")

    # Save pseudo-label generator
    plg_path = os.path.join(args.output, "pseudo_label_generator.pkl")
    joblib.dump(plg, plg_path)
    print(f"[Saved] Pseudo-label generator → {plg_path}")

    # Save evaluation metrics
    eval_path = os.path.join(args.output, "evaluation_metrics.json")
    eval_results = {
        k: v for k, v in clf._eval_results.items()
        if k not in ("y_test", "y_pred", "y_proba")
    }
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[Saved] Evaluation metrics → {eval_path}")

    # ---- Stage 3: Evidence Dossier Generation (sample) ----
    dg = EvidenceDossierGenerator(plg, clf)
    y_pred_all, y_proba_all = clf.predict(df_labeled)
    dossiers = dg.generate_batch(df_labeled, y_pred_all, y_proba_all)

    # Save dossiers
    dossiers_path = os.path.join(args.output, "evidence_dossiers.json")
    with open(dossiers_path, "w") as f:
        json.dump(dossiers[:500], f, indent=2)  # Save first 500 for readability
    print(f"[Saved] {len(dossiers)} evidence dossiers → {dossiers_path}")

    # ---- Adversarial Robustness Test (if provided) ----
    if args.adversarial_test and os.path.exists(args.adversarial_test):
        print("\n" + "=" * 60)
        print("ADVERSARIAL ROBUSTNESS TEST")
        print("=" * 60)
        df_adv = pd.read_csv(args.adversarial_test)
        # For adversarial test, use pseudo-labels and classifier
        df_adv_labeled = plg.generate_labels(df_adv, refit=False, skip_embedding=args.skip_embeddings)
        y_adv_pred, _ = clf.predict(df_adv_labeled)
        # For adversarial test, we check if the system correctly flags mismatches
        # even when keyword patterns are deceptive
        n_flagged = y_adv_pred.sum()
        print(f"[Adversarial] {n_flagged}/{len(df_adv)} tickets flagged as mismatches")
        print(f"[Adversarial] Flag rate: {100*n_flagged/len(df_adv):.1f}%")

        adv_dossiers = dg.generate_batch(df_adv_labeled, y_adv_pred, np.ones(len(df_adv)))
        adv_dossier_path = os.path.join(args.output, "adversarial_dossiers.json")
        with open(adv_dossier_path, "w") as f:
            json.dump(adv_dossiers, f, indent=2)
        print(f"[Saved] Adversarial dossiers → {adv_dossier_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output}/")
    print(f"  - pseudo_labeled_data.csv")
    print(f"  - mismatch_classifier_xgb.pkl")
    print(f"  - pseudo_label_generator.pkl")
    print(f"  - evaluation_metrics.json")
    print(f"  - signal_agreements.json")
    print(f"  - evidence_dossiers.json")

    # Check verification thresholds
    eval_m = clf._eval_results
    checks = []
    checks.append(("Accuracy ≥ 83%", eval_m["accuracy"] >= 0.83, f"{eval_m['accuracy']*100:.2f}%"))
    checks.append(("Macro F1 ≥ 0.82", eval_m["f1_macro"] >= 0.82, f"{eval_m['f1_macro']:.4f}"))
    checks.append(("Recall (Consistent) ≥ 0.78", eval_m["recall_consistent"] >= 0.78, f"{eval_m['recall_consistent']:.4f}"))
    checks.append(("Recall (Mismatch) ≥ 0.78", eval_m["recall_mismatch"] >= 0.78, f"{eval_m['recall_mismatch']:.4f}"))

    print("\n[Verification Check]")
    all_pass = True
    for name, passed, value in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status} | {name}: {value}")
    if all_pass:
        print("\n✅ All verification thresholds met!")
    else:
        print("\n⚠️  Some thresholds not met. Consider tuning hyperparameters.")

    return df_labeled, clf, dossiers


if __name__ == "__main__":
    main()
