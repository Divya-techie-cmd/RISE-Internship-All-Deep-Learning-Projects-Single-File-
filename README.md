# SMART HR – Employee Attrition Prediction and Workforce Intelligence

A single-file Streamlit project for educational HR analytics. It uploads CSV/XLSX data, performs preprocessing, trains Logistic Regression and Random Forest (plus XGBoost when installed), compares holdout metrics, explores workforce patterns, estimates risk, and exports CSV reports.

> **Responsible use:** This is a classroom decision-support demo, not an employment decision tool. Probabilities are uncertain estimates, are not causal, and must not be used to make hiring, firing, promotion, compensation, or other employment decisions. The app excludes age, gender, marital status, and direct identifiers from model features. It does not claim production fairness, calibration, or external validation.

## Run locally

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a CSV or XLSX file with an `Attrition` column. IBM's commonly used sample labels it `Yes` and `No`; the app supports those and `True/False` or `1/0`. It will not fabricate results before you provide data and click **Train / refresh models**.

## Dataset

Use the IBM HR Analytics Employee Attrition & Performance sample dataset, obtained from its original publisher or an authorized educational source. Review that source's terms and remove direct identifiers before uploading. The application expects one row per employee and an `Attrition` target. It drops known identifiers/constants and excludes `Age`, `Gender`, and `MaritalStatus` from training. Other columns are imputed/encoded inside an sklearn pipeline, fitted only on the training partition.

## Included pages

- Overview KPIs and department/overtime charts
- Employee analytics with department, role, gender (descriptive only), overtime, and age filters
- Individual input risk estimate, risk band, model-wide importance, and human-reviewed discussion prompts
- Global permutation explainability
- Model comparison, confusion matrix, ROC curve, accuracy/precision/recall/F1/ROC-AUC
- Searchable employee table and CSV downloads

## Modeling notes

A stratified 75/25 train-test split is used with a fixed seed. Imputation, one-hot encoding, and numeric scaling are fit within a pipeline on training data to reduce leakage. Class weights address imbalance; no synthetic oversampling is used. Optional RandomizedSearchCV tunes a Random Forest using 3-fold CV on training data. XGBoost is included when its package is available. The holdout is a single-split educational estimate, not a production evaluation. SHAP, login/authentication, PostgreSQL, PDF output, and a multi-user audit trail are **not implemented** in this single-file Streamlit edition.

## Deployment (Streamlit Community Cloud)

This assistant cannot execute Streamlit or provision an external deployment. To publish it: push this branch to GitHub, sign in at [share.streamlit.io](https://share.streamlit.io/), choose **Create app**, select this repository and branch `feature/smart-hr-attrition`, set the main file to `app.py`, and deploy. Cloud install uses `requirements.txt`. Keep employee datasets private; do not commit personal data or secrets. Deployment requires your account authorization and should use an access-controlled environment for sensitive data.

## Architecture

```text
CSV/XLSX upload -> validation/clean target -> stratified split
   -> sklearn preprocessing pipeline (impute / scale / one-hot)
      -> Logistic Regression / Random Forest / optional XGBoost
         -> holdout metrics + permutation importance
            -> Streamlit pages: overview, analytics, prediction, explainability, reports
```

## Project report (brief)

**Problem:** HR teams need descriptive analysis of historical attrition and a way to inspect patterns without treating a classifier as certainty.

**Method:** Validate target labels; exclude direct identifiers and selected sensitive demographic fields; stratify the split; fit preprocessing only within training folds; compare class-weighted classifiers on held-out data; report multiple metrics and confusion/ROC views. Optional randomized search is confined to the training set.

**Limitations:** Small or biased historical datasets, proxy variables, nonrepresentative splits, changes in workforce context, and uncalibrated probabilities can make estimates unreliable. Feature importance is association-based, not causal. This prototype does not implement group fairness testing, probability calibration, authentication, or deployment-grade data controls. Human review, legal/privacy review, and validation are required before any operational use.

## Viva questions and short answers

1. **Why stratified splitting?** It keeps the class proportion approximately consistent in train and test partitions.
2. **How is leakage reduced?** Imputation, scaling, and encoding are fitted inside a pipeline after splitting; optional tuning uses CV on training data only.
3. **Why use several metrics?** Accuracy can hide minority-class errors; precision/recall/F1 and ROC-AUC provide complementary views.
4. **Why class weights?** They penalize minority-class errors more without synthesizing employee records.
5. **What does ROC-AUC mean?** Ranking ability across thresholds, not a guarantee of calibrated probabilities or useful decisions.
6. **Are feature importances causal?** No. Permutation importance measures predictive dependence on a holdout sample, not cause.
7. **Why exclude protected attributes?** To avoid direct use of selected sensitive attributes; this does not remove proxy bias, so subgroup audits remain necessary.
8. **Can this decide who will leave?** No. It estimates patterns in historical data and is not suitable for individual employment decisions.
9. **How is categorical data handled?** Missing values are imputed and categories one-hot encoded with unseen categories ignored.
10. **What should production work add?** Authentication, role permissions, secure storage, audit logs, calibration, temporal/external validation, fairness monitoring, consent/privacy governance, and human oversight.
