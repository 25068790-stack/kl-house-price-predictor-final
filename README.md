# Kuala Lumpur House Price Predictor — Final Deployment

This is the independent final Streamlit deployment package for the master's
research prototype.

## Repository structure

- `app.py` — Streamlit interface
- `predictor.py` — portable frozen preprocessing and prediction logic
- `xgboost_booster.ubj` — final fitted XGBoost booster
- `deployment_config.json` — frozen preprocessing state, categories and metrics
- `requirements.txt` — deployment dependencies
- `.streamlit/config.toml` — app theme
- `verification_report.json` — export-equivalence verification
- `verify_package.py` — quick smoke test

## Why the original joblib file is not used directly

The research pipeline contains a notebook-defined `FunctionTransformer`.
Direct joblib deployment would therefore depend on the exact notebook namespace
and exact scikit-learn serialization environment.

For deployment only, the already-fitted preprocessing state was extracted and
reproduced explicitly, and the already-fitted XGBoost booster was exported to
XGBoost's stable model format.

No retraining, hyperparameter tuning, or model reselection was performed during
this serving export.

## Frozen research evaluation

- Hold-out RMSE: RM 661,168.89
- Hold-out MAE: RM 286,596.10
- Hold-out MAPE: 14.6989%
- Hold-out R²: 0.8934
- Bootstrap resampling unit: Predictor-profile group

These values refer to the independent hold-out evaluation, not to the
post-evaluation refitted deployment model.

## Deployment

Upload the contents of this folder to the root of a new GitHub repository, then
create a new Streamlit Community Cloud app with `app.py` as the entry point.

Do not upload or use the older deployment joblib files for this final app.
