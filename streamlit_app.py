"""
Support Integrity Auditor (SIA) — Streamlit Web App
=====================================================
MARS Open Projects 2026
Praneshwar Kannan Kommiya | 23117102 | B.Tech ME 4th Year

Features:
  - Single-ticket form input → binary judgment + full Evidence Dossier
  - Batch CSV upload → predictions + dossiers
  - Priority Mismatch Dashboard with distribution charts
  - Severity delta heatmap across categories and channels

Usage:
    streamlit run streamlit_app.py
"""

import json
import os
import sys
import warnings
from io import StringIO

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit command
# ---------------------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load artifacts (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model and pseudo-label generator."""
    from train_pipeline import MismatchClassifier, PseudoLabelGenerator
    
    model_dir = "models"
    clf_path = os.path.join(model_dir, "mismatch_classifier_xgb.pkl")
    plg_path = os.path.join(model_dir, "pseudo_label_generator.pkl")

    if not os.path.exists(clf_path) or not os.path.exists(plg_path):
        st.error("⚠️ Trained model not found! Please run `train_pipeline.py` first.")
        st.stop()

    artifacts = joblib.load(clf_path)
    
    # Reconstruct classifier from components
    clf = MismatchClassifier(random_state=42)
    clf.model = artifacts["xgb_model"]
    clf.tfidf_vectorizer = artifacts["tfidf_vectorizer"]
    clf.label_encoder_channel = artifacts["label_encoder_channel"]
    clf.label_encoder_category = artifacts["label_encoder_category"]
    clf.scaler = artifacts["scaler"]
    clf.feature_names_ = artifacts.get("feature_names", [])
    clf._fitted = True
    
    plg = joblib.load(plg_path)
    return clf, plg


# ---------------------------------------------------------------------------
# Helper: Run inference on a DataFrame
# ---------------------------------------------------------------------------

def run_inference(df: pd.DataFrame, clf, plg):
    """Run pseudo-labeling and classification on a DataFrame."""
    from train_pipeline import EvidenceDossierGenerator

    # Generate pseudo-labels using pre-fitted models (refit=False for inference)
    df_labeled = plg.generate_labels(df, refit=False)

    # Classify
    y_pred, y_proba = clf.predict(df_labeled)

    # Generate dossiers
    dg = EvidenceDossierGenerator(plg, clf)
    dossiers = dg.generate_batch(df_labeled, y_pred, y_proba)

    df_labeled["predicted_mismatch"] = y_pred
    df_labeled["mismatch_confidence"] = y_proba.round(4)

    return df_labeled, dossiers


# ---------------------------------------------------------------------------
# UI: Single Ticket Form
# ---------------------------------------------------------------------------

def render_single_ticket(clf, plg):
    st.subheader("📝 Single Ticket Audit")

    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Ticket Subject", placeholder="e.g., Login failed - Account locked")
            priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"], index=1)
            channel = st.selectbox("Ticket Channel", ["Email", "Chat", "Web Form", "Phone", "Social Media"])
            category = st.selectbox("Issue Category", ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
        with col2:
            description = st.text_area("Ticket Description", placeholder="Describe the issue in detail...")
            resolution_time = st.slider("Resolution Time (hours)", 1, 120, 24)
            customer_email = st.text_input("Customer Email (optional)", placeholder="user@example.com")

        submitted = st.form_submit_button("🔍 Audit Ticket", type="primary", use_container_width=True)

    if submitted:
        if not subject.strip() or not description.strip():
            st.warning("Please fill in both Subject and Description.")
            return

        # Build single-row DataFrame
        ticket_data = {
            "Ticket_ID": ["SIA-MANUAL-001"],
            "Customer_Name": ["Manual Entry"],
            "Customer_Email": [customer_email or "user@example.com"],
            "Ticket_Subject": [subject],
            "Ticket_Description": [description],
            "Issue_Category": [category],
            "Priority_Level": [priority],
            "Ticket_Channel": [channel],
            "Submission_Date": ["2026-06-13"],
            "Resolution_Time_Hours": [resolution_time],
            "Assigned_Agent": ["N/A"],
            "Satisfaction_Score": [3],
        }
        df_single = pd.DataFrame(ticket_data)

        with st.spinner("Analyzing ticket..."):
            df_result, dossiers = run_inference(df_single, clf, plg)

        row = df_result.iloc[0]
        is_mismatch = bool(row["predicted_mismatch"])
        confidence = float(row["mismatch_confidence"])

        # Display result
        st.markdown("---")
        if is_mismatch:
            st.error(f"⚠️ **PRIORITY MISMATCH DETECTED** (confidence: {confidence:.2%})")
        else:
            st.success(f"✅ **Priority is Consistent** (confidence: {confidence:.2%})")

        # Show key details
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Assigned Priority", row["Priority_Level"])
        c2.metric("Inferred Severity", row["inferred_severity"])
        c3.metric("Mismatch Type", row["mismatch_type"])
        c4.metric("Severity Delta", str(int(row["severity_delta"])))

        # Show dossier if mismatch
        if is_mismatch and dossiers:
            dossier = dossiers[0]
            st.markdown("### 📋 Evidence Dossier")
            st.json(dossier)

            st.markdown("#### 🔬 Feature Evidence Breakdown")
            for item in dossier.get("feature_evidence", []):
                signal_name = item.get("signal", "unknown").upper()
                source = item.get("source_field", "N/A")
                with st.expander(f"📌 {signal_name} — source: `{source}`"):
                    for k, v in item.items():
                        if k not in ("signal", "source_field"):
                            st.write(f"**{k}:** {v}")

            st.markdown("#### 📝 Explanation")
            st.info(dossier.get("constraint_analysis", "No explanation available."))


# ---------------------------------------------------------------------------
# UI: Batch CSV Upload
# ---------------------------------------------------------------------------

def render_batch_upload(clf, plg):
    st.subheader("📁 Batch CSV Upload")

    st.markdown("""
    Upload a CSV file with the following columns (matching the training data format):
    `Ticket_ID`, `Customer_Name`, `Customer_Email`, `Ticket_Subject`, `Ticket_Description`,
    `Issue_Category`, `Priority_Level`, `Ticket_Channel`, `Submission_Date`,
    `Resolution_Time_Hours`, `Assigned_Agent`, `Satisfaction_Score`
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df)}** tickets. Preview:")
            st.dataframe(df.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Audit", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} tickets..."):
                    df_result, dossiers = run_inference(df, clf, plg)

                n_flagged = int(df_result["predicted_mismatch"].sum())
                st.success(f"✅ Processed {len(df)} tickets — **{n_flagged} mismatches detected**")

                # Download buttons
                csv_data = df_result.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="sia_predictions.csv",
                    mime="text/csv",
                )

                dossiers_json = json.dumps(dossiers, indent=2)
                st.download_button(
                    "📥 Download Evidence Dossiers (JSON)",
                    data=dossiers_json,
                    file_name="sia_dossiers.json",
                    mime="application/json",
                )

                # Show flagged tickets
                if n_flagged > 0:
                    st.markdown("### 🚩 Flagged Tickets")
                    flagged = df_result[df_result["predicted_mismatch"] == 1]
                    st.dataframe(
                        flagged[["Ticket_ID", "Ticket_Subject", "Priority_Level",
                                  "inferred_severity", "mismatch_type", "severity_delta",
                                  "mismatch_confidence"]].head(20),
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ---------------------------------------------------------------------------
# UI: Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(clf, plg):
    st.subheader("📊 Priority Mismatch Dashboard")

    # Load the pseudo-labeled training data for dashboard stats
    data_path = "models/pseudo_labeled_data.csv"
    if not os.path.exists(data_path):
        st.warning("No training results found. Run `train_pipeline.py` first to generate dashboard data.")
        return

    df_full = pd.read_csv(data_path)

    # Compute predictions on full dataset (use cached copy if available)
    pred_path = "models/full_predictions.csv"
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
    else:
        # Run on a sample for dashboard speed
        df_sample = df_full.sample(n=min(3000, len(df_full)), random_state=42)
        df_pred, _ = run_inference(df_sample, clf, plg)
        df_pred.to_csv(pred_path, index=False)

    # ---- Row 1: Key Metrics ----
    total = len(df_pred)
    n_mismatch = int(df_pred["predicted_mismatch"].sum())
    n_hidden = int((df_pred["mismatch_type"] == "Hidden Crisis").sum())
    n_false = int((df_pred["mismatch_type"] == "False Alarm").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tickets Audited", f"{total:,}")
    m2.metric("Mismatches Flagged", f"{n_mismatch:,}", f"{100*n_mismatch/total:.1f}%")
    m3.metric("Hidden Crises", n_hidden)
    m4.metric("False Alarms", n_false)

    # ---- Row 2: Distribution Charts ----
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Mismatch Distribution by Priority")
        try:
            import plotly.express as px
            mismatch_by_priority = df_pred.groupby("Priority_Level")["predicted_mismatch"].agg(["sum", "count"])
            mismatch_by_priority["rate"] = mismatch_by_priority["sum"] / mismatch_by_priority["count"]
            fig1 = px.bar(
                mismatch_by_priority.reset_index(),
                x="Priority_Level", y="rate",
                title="Mismatch Rate by Assigned Priority",
                labels={"rate": "Mismatch Rate", "Priority_Level": "Assigned Priority"},
                color="Priority_Level",
                color_discrete_map={"Low": "green", "Medium": "blue", "High": "orange", "Critical": "red"},
            )
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart error: {e}")

    with c2:
        st.markdown("#### Mismatch Type Split")
        try:
            type_counts = df_pred["mismatch_type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig2 = px.pie(
                type_counts, names="Type", values="Count",
                title="Consistent vs Mismatch Breakdown",
                color="Type",
                color_discrete_map={"Consistent": "#4CAF50", "Hidden Crisis": "#FF9800", "False Alarm": "#F44336"},
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart error: {e}")

    # ---- Row 3: Severity Delta Heatmap ----
    st.markdown("---")
    st.markdown("#### Severity Delta Heatmap — Categories × Channels")

    try:
        heatmap_data = df_pred.pivot_table(
            values="severity_delta",
            index="Issue_Category",
            columns="Ticket_Channel",
            aggfunc="mean",
        ).round(2)

        fig3 = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            title="Mean Severity Delta (Inferred − Assigned) per Category & Channel",
            labels={"x": "Channel", "y": "Category", "color": "Severity Delta"},
            color_continuous_scale="RdBu_r",
            range_color=[-2, 2],
        )
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.warning(f"Heatmap error: {e}")

    # ---- Row 4: Top Contributing Signals ----
    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Top Signals for Hidden Crisis")
        hidden = df_pred[df_pred["mismatch_type"] == "Hidden Crisis"]
        if len(hidden) > 0:
            cat_counts = hidden["Issue_Category"].value_counts()
            fig4 = px.bar(
                x=cat_counts.index, y=cat_counts.values,
                title="Hidden Crisis — Category Breakdown",
                labels={"x": "Category", "y": "Count"},
                color=cat_counts.index,
            )
            fig4.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.markdown("#### Top Signals for False Alarm")
        false_alarm = df_pred[df_pred["mismatch_type"] == "False Alarm"]
        if len(false_alarm) > 0:
            cat_counts_fa = false_alarm["Issue_Category"].value_counts()
            fig5 = px.bar(
                x=cat_counts_fa.index, y=cat_counts_fa.values,
                title="False Alarm — Category Breakdown",
                labels={"x": "Category", "y": "Count"},
                color=cat_counts_fa.index,
            )
            fig5.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig5, use_container_width=True)

    # ---- Row 5: Resolution Time vs Severity Delta ----
    st.markdown("---")
    st.markdown("#### Resolution Time vs Severity Delta")
    try:
        df_viz = df_pred.sample(n=min(2000, len(df_pred)), random_state=42)
        fig6 = px.scatter(
            df_viz, x="Resolution_Time_Hours", y="severity_delta",
            color="mismatch_type",
            hover_data=["Ticket_Subject", "Priority_Level", "inferred_severity"],
            title="Resolution Time vs Severity Delta (bubble = confidence)",
            size="mismatch_confidence",
            color_discrete_map={"Consistent": "#4CAF50", "Hidden Crisis": "#FF9800", "False Alarm": "#F44336"},
            opacity=0.6,
        )
        fig6.update_layout(height=400)
        st.plotly_chart(fig6, use_container_width=True)
    except Exception as e:
        st.warning(f"Scatter error: {e}")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    # Sidebar
    st.sidebar.title("🔍 SIA")
    st.sidebar.markdown("### Support Integrity Auditor")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**MARS Open Projects 2026**")
    st.sidebar.markdown("Praneshwar Kannan Kommiya")
    st.sidebar.markdown("23117102 | B.Tech ME 4th Year")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "📝 Single Ticket Audit", "📁 Batch Upload"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About SIA:**
    
    SIA detects priority mismatches in CRM support tickets using:
    - 🔤 NLP keyword analysis
    - 🧠 Semantic embeddings  
    - ⏱️ Resolution time patterns
    - 🤖 XGBoost classifier
    
    Every evidence claim is traceable to specific ticket fields — zero hallucinations.
    """)

    # Load model
    try:
        clf, plg = load_artifacts()
    except Exception:
        st.warning("Model not found. Please train the pipeline first.")
        return

    # Render selected page
    if page == "📊 Dashboard":
        st.title("📊 Priority Mismatch Dashboard")
        st.caption("Overview of mismatch patterns across the ticket ecosystem")
        render_dashboard(clf, plg)

    elif page == "📝 Single Ticket Audit":
        st.title("📝 Single Ticket Audit")
        st.caption("Submit one ticket and get an instant priority integrity check")
        render_single_ticket(clf, plg)

    elif page == "📁 Batch Upload":
        st.title("📁 Batch CSV Upload")
        st.caption("Upload a batch of tickets for bulk auditing")
        render_batch_upload(clf, plg)

    # Footer
    st.markdown("---")
    st.caption("SIA — Support Integrity Auditor | MARS Open Projects 2026 | Built with ❤️ using Streamlit")


if __name__ == "__main__":
    main()
