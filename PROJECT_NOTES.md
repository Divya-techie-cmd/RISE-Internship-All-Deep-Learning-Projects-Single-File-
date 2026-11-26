# SMART HR system – project notes

## Purpose and scope
Educational workforce analytics and decision support using historical attrition labels. The single-file implementation is `app.py`; it is designed to run locally or on Streamlit Community Cloud after the owner authorizes deployment.

## Data schema
One employee per row. Required label: `Attrition` (`Yes`/`No`, `True`/`False`, or `1`/`0`). Other feature columns are inferred. Known IDs and constants, age, gender, and marital status are excluded from model inputs. No employee dataset is committed in this repository.

## API / database
This Streamlit edition has no HTTP API or database. It operates on an uploaded file in the current app session and exports CSV. PostgreSQL/FastAPI are intentionally outside this runnable single-file scope and should be added only with authentication, authorization, privacy controls, and auditability.

## Validation and safety
Stratified holdout; preprocessing pipeline; class-weighted models; optional training-only CV; no fabricated metrics. Risk labels use configurable demonstration cutoffs (25% and 60%) and should not be interpreted as validated organizational thresholds. Outputs must not be used for automated employment decisions.
