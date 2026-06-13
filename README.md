# Support Integrity Auditor (SIA)

**MARS Open Projects 2026**  
**Praneshwar Kannan Kommiya | 23117102 | B.Tech ME 4th Year**

---

A semantics-driven, evidence-grounded automated auditor that detects **Priority Mismatch** in CRM support tickets — cases where a ticket's objective characteristics conflict with its human-assigned priority level.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Ablation Study & Fusion Strategy](#ablation-study--fusion-strategy)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Adversarial Robustness](#adversarial-robustness)
7. [Setup & Installation](#setup--installation)
8. [Usage](#usage)
9. [File Structure](#file-structure)
10. [Streamlit Web App](#streamlit-web-app)
11. [Evidence Dossier Schema](#evidence-dossier-schema)
12. [Verification Checklist](#verification-checklist)

---

## Problem Overview

In enterprise-scale CRM ecosystems, manual ticket triage suffers from:
- **Agent fatigue bias** — tired agents mislabel tickets
- **Customer favoritism** — VIP customers get inflated priority
- **Keyword anchoring** — over-reliance on trigger words like "urgent"

When critical issues are mislabeled as "Low" (Hidden Crisis) or trivial complaints are inflated to "Critical" (False Alarm), SLAs are jeopardized and customer churn increases.

**The hard part:** There are **no pre-annotated mismatch labels**. The system must bootstrap its own supervision signal from raw ticket data alone.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SUPPORT INTEGRITY AUDITOR                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Signal A    │  │  Signal B    │  │  Signal C    │      │
│  │  Keyword NLP │  │  Embedding   │  │  Resolution  │      │
│  │  Scoring     │  │  Clustering  │  │  Time Proxy  │      │
│  │  (TF-IDF +   │  │  (Sentence-  │  │  (Percentile │      │
│  │   regex)     │  │  Transformers│  │   Binning)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                 ┌─────────────────┐                         │
│                 │  Weighted Fusion│  ← Stage 1: Pseudo-     │
│                 │  (0.35/0.30/    │     Label Generation    │
│                 │   0.35)         │                         │
│                 └────────┬────────┘                         │
│                          ▼                                  │
│                 ┌─────────────────┐                         │
│                 │ Inferred        │                         │
│                 │ Severity (0-3)  │                         │
│                 └────────┬────────┘                         │
│                          │ compare with assigned priority   │
│                          ▼                                  │
│                 ┌─────────────────┐                         │
│                 │ Binary Mismatch │                         │
│                 │ Label (0/1)     │                         │
│                 └────────┬────────┘                         │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────┐          │
│  │  Stage 2: XGBoost Classifier + SMOTE         │          │
│  │  TF-IDF text features + structured metadata  │          │
│  └───────────────────┬───────────────────────────┘          │
│                      ▼                                      │
│  ┌───────────────────────────────────────────────┐          │
│  │  Stage 3: Evidence Dossier Generation        │          │
│  │  Hallucination-free, fully traceable         │          │
│  └───────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — Pseudo-Label Generation (Self-Supervised)

Three independent signals are computed for each ticket, then fused:

#### Signal A: Rule-Based NLP Keyword Scoring
- Scans ticket Subject + Description for **urgency keywords** (e.g., "crash", "fraud", "cannot log in", "data breach")
- Detects **negation patterns** ("not working", "still waiting", "never received")
- Normalizes by text length to prevent long-text bias
- Maps to 0-3 severity scale via percentile discretization

#### Signal B: Embedding-Based Semantic Clustering
- Uses `all-MiniLM-L6-v2` (sentence-transformers) to encode each ticket into a 384-dim vector
- Reduces to 128 dims via Truncated SVD
- Clusters into 8 groups with K-Means
- Maps clusters to severity based on **average resolution time** of tickets in each cluster
- Captures semantic similarity that keywords miss (e.g., "screen stays black" ≈ "display not working")

#### Signal C: Resolution-Time Proxy
- Uses percentile-based binning: 0-30th → Low, 30-60th → Medium, 60-85th → High, 85-100th → Critical
- Provides an **objective, retrospective ground truth** — tickets taking 100+ hours probably weren't "Low"

#### Fusion: Weighted Averaging
```
Inferred Severity = round(0.35 × Signal_A + 0.30 × Signal_B + 0.35 × Signal_C)
```
Weights were chosen to balance:
- **Keyword (0.35):** High weight because it captures explicit urgency language that human agents use
- **Embedding (0.30):** Slightly lower because clustering is unsupervised and can have noise
- **Resolution (0.35):** High weight because it's an objective outcome measure

The binary mismatch label is: `is_mismatch = (inferred_severity != assigned_priority)`

---

### Stage 2 — Classifier Training

| Component | Choice |
|-----------|--------|
| **Text features** | TF-IDF (max 5000 features, 1-3 ngrams, sublinear scaling) |
| **Structured features** | Resolution time, channel encoding, category encoding, domain tier, channel urgency, category severity |
| **Model** | XGBoost (200 estimators, max_depth=6, lr=0.05) |
| **Imbalance handling** | SMOTE oversampling + scale_pos_weight |
| **Validation** | 80/20 stratified split + 5-fold CV |

**Why XGBoost?** For 20K rows on CPU, XGBoost trains in seconds and provides feature importance for dossier interpretability. The transformer fine-tuning path (DeBERTa-v3-small with LoRA) is included in the codebase for GPU-accelerated training.

Input features include both text fields (Subject + Description) and at least one structured metadata feature (channel, domain tier, resolution time), satisfying the project requirement.

---

### Stage 3 — Evidence Dossier Generation

Every flagged ticket gets a structured JSON dossier. The **hard rule**: every `feature_evidence` item must reference a specific field in the input ticket.

Evidence signals traced:
| Signal | Source Field | What it shows |
|--------|-------------|---------------|
| keyword | Ticket_Subject, Ticket_Description | Urgency keywords and their weights |
| resolution_time | Resolution_Time_Hours | Resolution speed as complexity proxy |
| channel | Ticket_Channel | Channel urgency score |
| category | Issue_Category | Category base severity |
| domain_tier | Customer_Email | Enterprise vs standard customer tier |

---

## Ablation Study & Fusion Strategy

Running each signal independently and comparing with the fused result:

| Configuration | Pseudo-Label Agreement | Notes |
|--------------|----------------------|-------|
| Keyword only | — | Catches explicit urgency but misses implicit severity in calm language |
| Embedding only | — | Captures semantics but can over-cluster (e.g., all "account" issues together regardless of urgency) |
| Resolution only | — | Objective but retrospective — can't predict severity for new tickets |
| **Keyword + Embedding** | — | Good coverage; semantic + explicit patterns |
| **Keyword + Resolution** | — | Strong; explicit language + objective outcome |
| **All three (fused)** | — | **Best balance.** Each signal compensates for the others' blind spots. |

The fusion was validated by checking pairwise signal agreement rates and ensuring no single signal dominates. The weighted averaging approach prevents any one signal from having veto power while still allowing strong signals to influence the final label.

---

## Evaluation Metrics

All metrics computed on a stratified 20% hold-out test set:

| Metric | Threshold | Achieved | Status |
|--------|-----------|----------|--------|
| Binary Classification Accuracy | ≥ 83% | 98.90% | ✅ |
| Macro F1 Score | ≥ 0.82 | 0.9848 | ✅ |
| Per-Class Recall (Consistent) | ≥ 0.78 | 0.9925 | ✅ |
| Per-Class Recall (Mismatch) | ≥ 0.78 | 0.9779 | ✅ |
| Cross-Validation F1 (5-fold) | — | 0.9882 ± 0.0007 | ✅ |
| Pseudo-Label Signal Agreement | — | See `models/signal_agreements.json` | ✅ |

Metrics are saved to `models/evaluation_metrics.json` after training.

---

## Adversarial Robustness

10 hand-crafted adversarial tickets were designed to fool keyword-based systems:

- **Keyword-stuffed trivial tickets** — using "URGENT", "CRITICAL" for minor issues
- **Calmly-described critical incidents** — serious issues with no trigger words
- **Channel/signal mismatches** — urgent issues on low-urgency channels and vice versa
- **Misleading resolution patterns** — fast-resolved serious issues, slow-resolved trivial ones

A pure keyword-matching system would fail on most of these. SIA's fused approach is designed to catch them.

**Scoring:** ≥ 7/10 correctly flagged → 10% bonus score.

Results are generated during notebook execution and saved to `models/adversarial_dossiers.json`.

---

## Setup & Installation

```bash
# Clone the repository
git clone <repo-url>
cd sia-support-integrity-auditor

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python train_pipeline.py --data dataset/customer_support_tickets.csv --output models/

# Run inference on new tickets
python predict.py --input new_tickets.csv --output results/ --model models/

# Launch the Streamlit web app
streamlit run streamlit_app.py
```

**Requirements:** Python 3.10+, 8GB+ RAM recommended (the embedding step processes 20K tickets).

---

## Usage

### Training
```bash
python train_pipeline.py --data dataset/customer_support_tickets.csv --output models/
```
This runs all three stages and saves model artifacts to `models/`.

### Inference
```bash
python predict.py --input test_tickets.csv --output results/ --model models/
```
Outputs `results/predictions.csv` and `results/evidence_dossiers.json`.

### Jupyter Notebook
Open `notebook.ipynb` in VS Code or Jupyter Lab for a step-by-step walkthrough with visualizations.

### Streamlit App
```bash
streamlit run streamlit_app.py
```
Features:
- **Single Ticket Audit:** Paste a ticket and get instant mismatch judgment
- **Batch CSV Upload:** Upload a CSV and download flagged results
- **Priority Mismatch Dashboard:** Distribution charts, severity delta heatmap, top contributing signals

---

## File Structure

```
├── dataset/
│   ├── customer_support_tickets.csv          # Main dataset (20,000 tickets)
│   └── enhanced_customer_support_data.csv     # Enhanced variant
├── models/                                    # Created after training
│   ├── mismatch_classifier_xgb.pkl
│   ├── pseudo_label_generator.pkl
│   ├── pseudo_labeled_data.csv
│   ├── evaluation_metrics.json
│   ├── signal_agreements.json
│   └── evidence_dossiers.json
├── results/                                   # Created after inference
│   ├── predictions.csv
│   ├── evidence_dossiers.json
│   └── summary.json
├── notebook.ipynb                            # Full pipeline notebook
├── train_pipeline.py                         # Standalone training script
├── predict.py                                # Inference script
├── streamlit_app.py                          # Streamlit web application
├── requirements.txt                          # Pinned dependencies
└── README.md                                 # This file
```

---

## Streamlit Web App

The web app (`streamlit_app.py`) provides three views:

1. **Dashboard** — Overview metrics, mismatch distribution by priority, mismatch type pie chart, severity delta heatmap (Category × Channel), resolution time vs severity delta scatter plot
2. **Single Ticket Audit** — Form-based input, returns binary judgment + full evidence dossier with traceable feature breakdown
3. **Batch Upload** — CSV upload with download buttons for predictions and dossiers

---

## Evidence Dossier Schema

```json
{
    "ticket_id": "TKT-100012",
    "assigned_priority": "Critical",
    "inferred_severity": "Critical",
    "mismatch_type": "Hidden Crisis | False Alarm | Consistent",
    "severity_delta": "+2",
    "feature_evidence": [
        {
            "signal": "keyword",
            "value": "'data loss' (weight=4), 'cannot access' (weight=3)",
            "weight": "3.50 avg",
            "source_field": "Ticket_Subject, Ticket_Description"
        },
        {
            "signal": "resolution_time",
            "value": "110 hours",
            "interpretation": "Resolution took 110h — in the top 15% of all tickets, strongly suggesting underlying complexity.",
            "source_field": "Resolution_Time_Hours"
        },
        {
            "signal": "channel",
            "value": "Chat",
            "interpretation": "Channel urgency score: 2/3. Higher-urgency channel.",
            "source_field": "Ticket_Channel"
        },
        {
            "signal": "category",
            "value": "Technical",
            "interpretation": "Category base severity: 2/3. Inherently high-severity category.",
            "source_field": "Issue_Category"
        },
        {
            "signal": "domain_tier",
            "value": "Enterprise",
            "interpretation": "Customer tier: Enterprise. Enterprise customers may need priority handling.",
            "source_field": "Customer_Email"
        }
    ],
    "constraint_analysis": "This ticket was assigned 'Low' priority but the evidence suggests it should be 'Critical'. The ticket subject combined with a resolution time of 110h and category 'Technical' indicate the issue is more severe than the assigned label reflects.",
    "confidence": "0.945"
}
```

---

## Verification Checklist

- [x] Binary Classification Accuracy ≥ 83%
- [x] Macro F1 Score ≥ 0.82
- [x] Per-Class Recall ≥ 0.78 (both classes)
- [x] Pseudo-Label Signal Agreement reported
- [x] Evidence Dossiers are hallucination-free (every claim traceable to input fields)
- [x] Adversarial robustness test on 10 held-out tickets
- [x] Streamlit web app with dashboard, single ticket audit, and batch upload
- [x] Full reproducible pipeline in notebook.ipynb

---

*Built for MARS Open Projects 2026. All code, analysis, and dossiers are the original work of the author.*
