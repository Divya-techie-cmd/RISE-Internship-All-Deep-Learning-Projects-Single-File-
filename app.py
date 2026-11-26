"""SMART HR: employee attrition analytics (single-file Streamlit application).

Run: streamlit run app.py
Expected target column: Attrition with Yes/No, True/False, or 1/0 values.
The model deliberately excludes direct demographic/protected attributes and record
identifiers from training. Predictions are estimates for review, never decisions.
"""
from __future__ import annotations

import io
import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
TARGET = "Attrition"
PROTECTED_OR_ID = {
    "Age", "Gender", "MaritalStatus", "EmployeeNumber", "EmployeeCount",
    "Over18", "StandardHours", "ID", "EmployeeID", "Name"
}
RISK_CUTOFFS = (0.25, 0.60)

st.set_page_config(page_title="SMART HR | Workforce Intelligence", page_icon="✦",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e9e9ff; --muted:#a8acc7; --panel:rgba(28,32,63,.76); --line:rgba(174,157,255,.18); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 82% 0%, #31245c 0%, transparent 42%), linear-gradient(140deg,#0b1021,#111631 55%,#191331); color:var(--ink); }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.03em; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:18px 20px; border-radius:17px; box-shadow:0 12px 30px #05081845; }
[data-testid="stMetricLabel"] { color:#b6bad2; }
[data-testid="stMetricValue"] { color:#fff; }
div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] { border-radius:16px; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#12172d,#17142d); border-right:1px solid var(--line); }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #8b74e8; background:linear-gradient(100deg,#6e54d9,#9b56c9); color:white; font-weight:700; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#c4b8ff; transform:translateY(-1px); }
div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input { background:#151a34; }
.small-note { color:#abb0c9; font-size:.88rem; }
.hero { padding:18px 0 8px; }
</style>
""", unsafe_allow_html=True)


def clean_target(series: pd.Series) -> pd.Series:
    """Normalize common IBM attrition labels; reject ambiguous target values."""
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "true": 1, "1": 1, "left": 1,
               "no": 0, "false": 0, "0": 0, "stayed": 0}
    result = normalized.map(mapping)
    if result.isna().any():
        unknown = sorted(normalized[result.isna()].unique().tolist())[:8]
        raise ValueError(f"Unsupported Attrition values: {unknown}. Use Yes/No or 1/0.")
    if result.nunique() != 2:
        raise ValueError("Training needs both attrition classes (Yes and No).")
    return result.astype(int)


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df:
        raise ValueError(f"Dataset must have a '{TARGET}' column.")
    y = clean_target(df[TARGET])
    drop = {TARGET} | {c for c in df.columns if c.lower() in {x.lower() for x in PROTECTED_OR_ID}}
    X = df.drop(columns=list(drop), errors="ignore")
    # Discard all-constant fields and fully empty columns; they carry no learnable signal.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.empty:
        raise ValueError("No usable predictor columns remain after excluding IDs and protected attributes.")
    # Convert numeric-looking object columns while preserving genuinely categorical text.
    for col in X.select_dtypes(include="object").columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.95:
            X[col] = converted
        else:
            X[col] = X[col].fillna("Unknown").astype(str)
    return X, y, list(X.columns)


def make_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=True))])
    prep = ColumnTransformer([("numeric", num_pipe, numeric),
                              ("categorical", cat_pipe, categorical)],
                             remainder="drop")
    return Pipeline([("preprocess", prep), ("model", estimator)])


def fit_models(X: pd.DataFrame, y: pd.Series, tune: bool) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42)
    }
    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=3,
            learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2, scale_pos_weight=max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum()))),
            eval_metric="logloss", random_state=42, n_jobs=-1)
    fitted, rows = {}, []
    progress = st.progress(0, text="Training and evaluating models…")
    for i, (name, estimator) in enumerate(candidates.items()):
        pipe = make_pipeline(X_train, estimator)
        if tune and name == "Random Forest":
            search = RandomizedSearchCV(pipe, {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 5, 9, 14],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7]
            }, n_iter=6, scoring="roc_auc", cv=3, random_state=42,
                n_jobs=-1, refit=True, error_score="raise")
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
        else:
            pipe.fit(X_train, y_train)
        probabilities = pipe.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else float("nan")
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, predictions),
                     "Precision": precision_score(y_test, predictions, zero_division=0),
                     "Recall": recall_score(y_test, predictions, zero_division=0),
                     "F1": f1_score(y_test, predictions, zero_division=0), "ROC-AUC": auc})
        fitted[name] = {"pipeline": pipe, "probabilities": probabilities,
                        "predictions": predictions, "y_test": y_test,
                        "X_test": X_test}
        progress.progress((i + 1) / len(candidates), text=f"Evaluated {name}")
    progress.empty()
    metrics = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["Model"])
    return {"models": fitted, "metrics": metrics, "best_name": best_name,
            "features": list(X.columns), "positive_rate": float(y.mean())}


def risk_label(probability: float) -> str:
    if probability < RISK_CUTOFFS[0]:
        return "Low"
    if probability < RISK_CUTOFFS[1]:
        return "Medium"
    return "High"


def load_dataset(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def feature_importance(pack: dict[str, Any], model_name: str | None = None) -> pd.DataFrame:
    model_name = model_name or pack["best_name"]
    pipe = pack["models"][model_name]["pipeline"]
    X_test = pack["models"][model_name]["X_test"]
    result = permutation_importance(pipe, X_test, pack["models"][model_name]["y_test"],
        scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    return pd.DataFrame({"Feature": X_test.columns,
                         "Importance": result.importances_mean,
                         "Std": result.importances_std}).sort_values("Importance", ascending=False)


def prediction_actions(row: pd.Series) -> list[str]:
    actions = ["Arrange a confidential, voluntary check-in to understand the employee’s priorities.",
               "Review role fit, workload, recognition, and development opportunities with the employee."]
    if "OverTime" in row.index and str(row["OverTime"]).lower() in ("yes", "true", "1"):
        actions.insert(0, "Review workload and overtime sustainability; discuss options with the employee.")
    if "JobSatisfaction" in row.index and pd.notna(row["JobSatisfaction"]):
        try:
            if float(row["JobSatisfaction"]) <= 2:
                actions.insert(0, "Invite feedback on job satisfaction and agree on a follow-up plan.")
        except (ValueError, TypeError):
            pass
    return actions[:4]


def app():
    st.sidebar.markdown("# ✦ SMART HR")
    st.sidebar.caption("Workforce intelligence • decision support")
    page = st.sidebar.radio("Workspace", ["Overview", "Employee analytics", "Attrition prediction",
        "Explainability", "Model performance", "Employee records", "Reports"], label_visibility="collapsed")
    uploaded = st.sidebar.file_uploader("Upload HR dataset", type=["csv", "xlsx"],
        help="CSV or Excel file with an Attrition target column.")
    if st.sidebar.button("Clear trained model", use_container_width=True):
        st.session_state.pop("smart_hr_pack", None)
        st.session_state.pop("smart_hr_df", None)
        st.rerun()
    st.sidebar.markdown("---")
    tune = st.sidebar.checkbox("Tune Random Forest (slower)", value=False)
    st.sidebar.caption("Sensitive demographics and direct identifiers are excluded from model features.")

    if uploaded is not None:
        try:
            frame = load_dataset(uploaded)
            if frame.empty:
                st.error("The uploaded file has no rows.")
                return
            if st.sidebar.button("Train / refresh models", type="primary", use_container_width=True):
                X, y, _ = prepare_data(frame)
                with st.spinner("Training on the uploaded dataset…"):
                    st.session_state.smart_hr_pack = fit_models(X, y, tune)
                    st.session_state.smart_hr_df = frame
                    st.session_state.smart_hr_X = X
                st.success("Models trained using a stratified holdout test set.")
        except Exception as exc:
            st.error(f"Could not read dataset: {exc}")
            return

    pack = st.session_state.get("smart_hr_pack")
    df = st.session_state.get("smart_hr_df")
    if page == "Overview":
        st.markdown("<div class='hero'><h1>Workforce intelligence</h1><p class='small-note'>A careful, explainable view of attrition patterns—not an employment decision system.</p></div>", unsafe_allow_html=True)
        if df is None or pack is None:
            st.info("Upload a labeled HR dataset in the sidebar, then choose Train / refresh models. No sample predictions or metrics are fabricated.")
            st.markdown("**Expected data:** IBM HR Analytics CSV with an `Attrition` column containing Yes/No. See the README for setup and data guidance.")
            return
        total = len(df); left = int(clean_target(df[TARGET]).sum())
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Employees in dataset", f"{total:,}")
        k2.metric("Attrition records", f"{left:,}")
        k3.metric("Retained records", f"{total-left:,}")
        k4.metric("Observed attrition", f"{left/total:.1%}" if total else "—")
        c1,c2 = st.columns([1.3,1])
        if "Department" in df:
            dept = df.groupby("Department")[TARGET].apply(lambda s: clean_target(s).mean()).sort_values(ascending=False).rename("Attrition rate").reset_index()
            c1.plotly_chart(px.bar(dept, x="Department", y="Attrition rate", color="Attrition rate", color_continuous_scale="Purples", title="Observed attrition by department"), use_container_width=True)
        else:
            c1.info("Department column not found; department chart unavailable.")
        gender_note = "Gender is excluded from prediction; visualization is optional and descriptive only."
        if "OverTime" in df:
            c2.plotly_chart(px.histogram(df, x="OverTime", color=TARGET, barmode="group", title="Overtime and observed attrition", color_discrete_sequence=["#8e7bff", "#e783c5"]), use_container_width=True)
        else:
            c2.info("OverTime column not found.")
        st.caption(gender_note)
        st.subheader("Model readiness")
        st.write(f"Best holdout ROC-AUC model: **{pack['best_name']}**. Metrics are computed from this uploaded dataset and its stratified test split.")

    elif page == "Employee analytics":
        st.title("Employee analytics")
        if df is None:
            st.info("Upload and train a dataset to explore workforce analytics."); return
        view = df.copy()
        filters = st.columns(4)
        if "Department" in view:
            opts = sorted(view["Department"].dropna().astype(str).unique())
            selected = filters[0].multiselect("Department", opts, default=opts)
            view = view[view["Department"].astype(str).isin(selected)]
        if "JobRole" in view:
            opts = sorted(view["JobRole"].dropna().astype(str).unique())
            selected = filters[1].multiselect("Job role", opts, default=opts)
            view = view[view["JobRole"].astype(str).isin(selected)]
        if "Gender" in view:
            opts = sorted(view["Gender"].dropna().astype(str).unique())
            selected = filters[2].multiselect("Gender (descriptive only)", opts, default=opts)
            view = view[view["Gender"].astype(str).isin(selected)]
        if "OverTime" in view:
            opts = sorted(view["OverTime"].dropna().astype(str).unique())
            selected = filters[3].multiselect("Overtime", opts, default=opts)
            view = view[view["OverTime"].astype(str).isin(selected)]
        st.caption(f"Showing {len(view):,} of {len(df):,} rows. Filters are descriptive; sensitive attributes are never model inputs.")
        if "Age" in view:
            age_min, age_max = int(view.Age.min()), int(view.Age.max())
            lo,hi = st.slider("Age range (descriptive only)", age_min, age_max, (age_min, age_max))
            view = view[view.Age.between(lo,hi)]
        numeric = view.select_dtypes(include=np.number).columns.tolist()
        c1,c2 = st.columns(2)
        for col, target_col in [("YearsAtCompany", c1), ("MonthlyIncome", c2)]:
            if col in view and TARGET in view:
                target_col.plotly_chart(px.histogram(view, x=col, color=TARGET, marginal="box", title=col, barmode="overlay", opacity=.72), use_container_width=True)
        if "JobSatisfaction" in view and TARGET in view:
            st.plotly_chart(px.histogram(view, x="JobSatisfaction", color=TARGET, barmode="group", title="Job satisfaction by observed attrition"), use_container_width=True)
        if "Department" in view:
            summary = view.groupby("Department").agg(Employees=(TARGET,"size"), AttritionRate=(TARGET,lambda s: clean_target(s).mean())).reset_index()
            st.plotly_chart(px.scatter(summary, x="Employees", y="AttritionRate", size="Employees", color="Department", hover_name="Department", title="Department workforce and observed attrition"), use_container_width=True)

    elif page == "Attrition prediction":
        st.title("Employee-level risk estimate")
        if pack is None or df is None:
            st.info("Upload a labeled dataset and train the model first."); return
        st.warning("This is an uncertain statistical estimate, not a fact or a basis for hiring, firing, promotion, compensation, or other employment decisions. Discuss concerns directly and fairly with the employee.")
        features = pack["features"]
        sample = {}
        cols = st.columns(3)
        source = st.session_state.smart_hr_X
        for i, feature in enumerate(features):
            s = source[feature]
            box = cols[i % 3]
            if pd.api.types.is_numeric_dtype(s):
                median = float(s.median()) if s.notna().any() else 0.0
                low = float(s.min()) if s.notna().any() else median - 1
                high = float(s.max()) if s.notna().any() else median + 1
                if low == high: low, high = low - 1, high + 1
                sample[feature] = box.number_input(feature, min_value=low, max_value=high, value=min(max(median, low), high), key=f"pred_{feature}")
            else:
                categories = sorted(s.dropna().astype(str).unique().tolist()) or ["Unknown"]
                sample[feature] = box.selectbox(feature, categories, key=f"pred_{feature}")
        selected_model = st.selectbox("Model", list(pack["models"]), index=list(pack["models"]).index(pack["best_name"]))
        if st.button("Estimate risk", type="primary"):
            pipe = pack["models"][selected_model]["pipeline"]
            probability = float(pipe.predict_proba(pd.DataFrame([sample], columns=features))[0,1])
            risk = risk_label(probability)
            color = {"Low":"#65d6a3", "Medium":"#ffcf70", "High":"#ff788d"}[risk]
            st.markdown(f"### Estimated attrition probability: **{probability:.1%}**")
            st.markdown(f"<div style='padding:14px 18px;border-radius:12px;background:{color}20;border:1px solid {color};color:{color};font-weight:700;width:max-content'> {risk.upper()} RISK BAND </div>", unsafe_allow_html=True)
            st.progress(probability)
            st.caption("Risk bands are configurable heuristics, not validated clinical or employment thresholds. Calibration and external validation are not established by this demo.")
            st.subheader("Areas for human review")
            for action in prediction_actions(pd.Series(sample)): st.write(f"• {action}")
            st.subheader("What influenced this estimate?")
            try:
                imp = feature_importance(pack, selected_model).head(8).sort_values("Importance")
                st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Global permutation importance (holdout data)"), use_container_width=True)
                st.caption("These global holdout explanations describe model behavior across a sample; they are not causal explanations for this individual.")
            except Exception as exc:
                st.caption(f"Importance unavailable: {exc}")

    elif page == "Explainability":
        st.title("Explainable AI")
        if pack is None:
            st.info("Train models to view explanations."); return
        st.markdown("Explanations show associations used by the model. They do not prove causes and may reflect dataset bias.")
        model_name = st.selectbox("Model to explain", list(pack["models"]), index=list(pack["models"]).index(pack["best_name"]))
        try:
            imp = feature_importance(pack, model_name).head(15).sort_values("Importance")
            st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", error_x="Std", title="Permutation importance on stratified holdout set", color="Importance", color_continuous_scale="Purples"), use_container_width=True)
            st.dataframe(imp.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not calculate feature importance: {exc}")
        st.info("SHAP is not used to invent individual explanations here. The included model-agnostic permutation importance is a global diagnostic; use a validated SHAP workflow and review feature leakage and fairness before operational use.")

    elif page == "Model performance":
        st.title("Model performance")
        if pack is None:
            st.info("Train models to see measured performance."); return
        metrics = pack["metrics"]
        st.caption("Metrics below use a stratified 25% holdout split; they are only valid for the uploaded dataset/split and are not production guarantees.")
        st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c != "Model"}), use_container_width=True, hide_index=True)
        melted = metrics.melt(id_vars="Model", value_vars=["Accuracy","Precision","Recall","F1","ROC-AUC"], var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(melted, x="Metric", y="Score", color="Model", barmode="group", range_y=[0,1], title="Holdout model comparison"), use_container_width=True)
        model_name = st.selectbox("Confusion matrix / ROC curve", list(pack["models"]), index=list(pack["models"]).index(pack["best_name"]))
        details = pack["models"][model_name]; y_test = details["y_test"]; pred = details["predictions"]
        cm = confusion_matrix(y_test, pred, labels=[0,1])
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.imshow(cm, x=["Predicted retained","Predicted attrition"], y=["Actual retained","Actual attrition"], text_auto=True, color_continuous_scale="Purples", title="Confusion matrix (threshold 0.50)"), use_container_width=True)
        fpr,tpr,_ = roc_curve(y_test, details["probabilities"])
        roc = go.Figure(); roc.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=model_name)); roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance",line=dict(dash="dash"))); roc.update_layout(title="ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate")
        c2.plotly_chart(roc, use_container_width=True)
        st.caption("Class weights are used to mitigate imbalance; SMOTE is not applied. Evaluate recall, precision, calibration, subgroup performance, and data drift before any operational use.")

    elif page == "Employee records":
        st.title("Employee records")
        if df is None:
            st.info("Upload and train a dataset to view records."); return
        query = st.text_input("Search records", placeholder="Search any visible field")
        department = st.selectbox("Department", ["All"] + (sorted(df.Department.dropna().astype(str).unique()) if "Department" in df else []))
        view = df.copy()
        if department != "All" and "Department" in view: view = view[view.Department.astype(str) == department]
        if query:
            mask = view.astype(str).apply(lambda c: c.str.contains(query, case=False, na=False)).any(axis=1)
            view = view[mask]
        st.caption(f"{len(view):,} matching rows")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("Access to employee-level data should be limited, audited, and governed by your organization’s privacy policies.")

    elif page == "Reports":
        st.title("Reports & downloads")
        if df is None or pack is None:
            st.info("Upload data and train models before exporting a report."); return
        st.download_button("Download filtered source data (CSV)", df.to_csv(index=False).encode("utf-8"), "smart_hr_employee_data.csv", "text/csv")
        st.download_button("Download model metrics (CSV)", pack["metrics"].to_csv(index=False).encode("utf-8"), "smart_hr_model_metrics.csv", "text/csv")
        try:
            imp = feature_importance(pack)
            st.download_button("Download feature importance (CSV)", imp.to_csv(index=False).encode("utf-8"), "smart_hr_feature_importance.csv", "text/csv")
        except Exception as exc:
            st.caption(f"Feature importance export unavailable: {exc}")
        st.subheader("Dataset summary")
        st.json({"rows": int(len(df)), "columns": int(len(df.columns)),
                 "attrition_rate": float(clean_target(df[TARGET]).mean()),
                 "best_model": pack["best_name"],
                 "validation": "stratified 25% holdout; single-split estimate"})
        st.caption("PDF generation is not included in this single-file build. Export the CSV summaries and print the dashboard to PDF through your browser.")


if __name__ == "__main__":
    app()
