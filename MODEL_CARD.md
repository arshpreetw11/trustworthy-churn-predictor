# Model Card — Trustworthy Churn Predictor

## Intended use
- Predicts the probability that a telecom customer will churn, with the top 3 reasons (SHAP).
- **Advisory only:** a retention agent sees the score and reasons and decides whether to make an offer. The model never acts automatically.
- Not for pricing, credit, or any decision outside retention outreach.

## Data
- IBM Telco Customer Churn dataset: 7,043 customers, 21 columns (7,032 after dropping 11 rows with blank `TotalCharges`).
- Churn rate: 26.6%. One snapshot per customer, no timestamps.
- Split: stratified 60 / 20 / 20 (train / validation / test). A time-based split was not possible.
- `gender` and `SeniorCitizen` are **not** model inputs; they are used only for fairness checks and to choose the decision threshold.

## Model
- XGBoost (400 trees, depth 3, learning rate 0.03), compared against logistic regression and LightGBM.
- Engineered features: average monthly spend, number of services, log of total charges.

## Performance (test set, 1,407 customers)
| Metric | Value |
|---|---|
| ROC-AUC | 0.85 |
| Recall in top 30% riskiest | 0.70 |
| Precision / Recall (per-group thresholds) | 0.487 / 0.826 |
| Brier score | 0.135 |

## Fairness
- Audited across `gender` and `SeniorCitizen`. Gender gap was small (FNR gap 0.047).
- With one shared threshold, seniors received false alarms at twice the rate of non-seniors (0.55 vs 0.27); equalized-odds gap 0.282.
- **Mitigation:** separate thresholds per group, each set to catch 80% of churners on validation data. Gap fell to 0.134.
- **Accepted trade-off:** fewer false alarms for seniors (0.55 → 0.43), but more senior churners missed (0.09 → 0.25). This is a policy choice and should be signed off by the business owner.

## Explainability
- Global top drivers (mean |SHAP|): Contract, tenure, OnlineSecurity, TechSupport, InternetService, AvgMonthlySpend.
- Every prediction comes with its top 3 SHAP reasons.

## Monitoring
- Population Stability Index (PSI) on input features; PSI > 0.2 triggers a review.
- Re-check fairness metrics on every retrain.

## Limitations
- Single public dataset; may not reflect any real company's customers.
- The senior group is small (~16%), so fairness numbers for seniors are noisy and vary between runs.
- Using a protected attribute to choose thresholds may be restricted by law in some regions.
- Results vary slightly by machine and library version.
