# ===== STEP 1: Load & clean =====
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
import os
df = pd.read_csv("telco.csv" if os.path.exists("telco.csv") else url)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).drop(columns=["customerID"]).reset_index(drop=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(df.shape, df["Churn"].mean().round(3))

SENSITIVE = ["gender", "SeniorCitizen"]
y = df["Churn"]
S = df[SENSITIVE].copy()
X = df.drop(columns=["Churn"] + SENSITIVE)

from sklearn.model_selection import train_test_split
X_tr, X_tmp, y_tr, y_tmp, S_tr, S_tmp = train_test_split(X, y, S, test_size=0.4, stratify=y, random_state=42)
X_val, X_te, y_val, y_te, S_val, S_te = train_test_split(X_tmp, y_tmp, S_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
print(len(X_tr), len(X_val), len(X_te))

# ===== STEP 2: Feature engineering =====
def add_features(d):
    d = d.copy()
    d["AvgMonthlySpend"] = d["TotalCharges"] / d["tenure"].clip(lower=1)
    svc = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
           "TechSupport","StreamingTV","StreamingMovies"]
    d["NumServices"] = (d[svc] == "Yes").sum(axis=1) + (d["InternetService"] != "No")
    d["TotalCharges"] = np.log1p(d["TotalCharges"])
    return d

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumServices"]
cat_cols = X.select_dtypes(exclude="number").columns.tolist()   # Contract, InternetService, ...
pre = Pipeline([
    ("fe", FunctionTransformer(add_features)),
    ("ct", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])),
])
pre.fit(X_tr)
feat_names = [n.split("__", 1)[1] for n in pre.named_steps["ct"].get_feature_names_out()]
Xt_tr, Xt_val, Xt_te = pre.transform(X_tr), pre.transform(X_val), pre.transform(X_te)

# Proxy-bias screen
from sklearn.feature_selection import mutual_info_classif
mi = pd.Series(mutual_info_classif(Xt_tr, S_tr["SeniorCitizen"], random_state=0), index=feat_names)
print("\nTop proxies for SeniorCitizen:\n", mi.sort_values(ascending=False).head(5).round(3))

# ===== STEP 3: Models =====
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.03, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1),
}
for name, m in models.items():
    m.fit(Xt_tr, y_tr)
    print(name, "val AUC:", round(roc_auc_score(y_val, m.predict_proba(Xt_val)[:, 1]), 4))
model = models["XGBoost"]

# ===== STEP 4: Fairness audit =====
from fairlearn.metrics import MetricFrame, false_negative_rate, equalized_odds_difference
p_val = model.predict_proba(Xt_val)[:, 1]
pred_val = (p_val >= 0.5).astype(int)
for s in SENSITIVE:
    mf = MetricFrame(metrics=false_negative_rate, y_true=y_val, y_pred=pred_val, sensitive_features=S_val[s])
    print(f"\nFNR by {s}:\n", mf.by_group.round(3), "\ngap:", round(mf.difference(), 3),
          " EO diff:", round(equalized_odds_difference(y_val, pred_val, sensitive_features=S_val[s]), 3))

# ===== STEP 5: Mitigation — per-group thresholds (on SeniorCitizen) =====
s_tr, s_val, s_te = S_tr["SeniorCitizen"].values, S_val["SeniorCitizen"].values, S_te["SeniorCitizen"].values
TARGET_RECALL = 0.80
p_val = model.predict_proba(Xt_val)[:, 1]

# (a) one threshold for everyone, chosen on validation to catch 80% of churners
single_t = np.quantile(p_val[y_val.values == 1], 1 - TARGET_RECALL)
# (b) one threshold PER GROUP, so each group gets 80% recall
group_t = {g: np.quantile(p_val[(s_val == g) & (y_val.values == 1)], 1 - TARGET_RECALL) for g in [0, 1]}
print("\nsingle threshold:", round(float(single_t), 3), "| per-group:", {g: round(float(t), 3) for g, t in group_t.items()})

from sklearn.metrics import precision_score, recall_score
from fairlearn.metrics import false_positive_rate
def predict_fair(p, s): return (p >= np.array([group_t[g] for g in s])).astype(int)

def compare(name, pred):
    fpr = MetricFrame(metrics=false_positive_rate, y_true=y_te, y_pred=pred, sensitive_features=s_te).by_group
    fnr = MetricFrame(metrics=false_negative_rate, y_true=y_te, y_pred=pred, sensitive_features=s_te).by_group
    print(f"{name:18s} EO-gap={equalized_odds_difference(y_te, pred, sensitive_features=s_te):.3f} "
          f"FPR non-senior/senior={fpr[0]:.2f}/{fpr[1]:.2f} FNR={fnr[0]:.2f}/{fnr[1]:.2f} "
          f"P={precision_score(y_te, pred):.3f} R={recall_score(y_te, pred):.3f}")

p_t = model.predict_proba(Xt_te)[:, 1]
compare("Single threshold", (p_t >= single_t).astype(int))
compare("Per-group", predict_fair(p_t, s_te))

# ===== STEP 6: Calibration =====
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(Xt_val, y_val)
p_te = cal.predict_proba(Xt_te)[:, 1]
print("\nBrier before:", round(brier_score_loss(y_te, p_t), 4), " after:", round(brier_score_loss(y_te, p_te), 4))

# ===== STEP 7: SHAP =====
import shap
explainer = shap.TreeExplainer(model)
sv = explainer(pd.DataFrame(Xt_te, columns=feat_names))
# group one-hot columns (Contract_Month-to-month, Contract_Two year...) back to "Contract"
orig = [next((c for c in cat_cols if f.startswith(c + "_")), f) for f in feat_names]
imp = pd.DataFrame(np.abs(sv.values), columns=orig).T.groupby(level=0).sum().T.mean().sort_values(ascending=False)
print("\nTop SHAP drivers (mean |SHAP|):\n", imp.head(6).round(3))
i = int(np.argmax(p_te))
top3 = pd.Series(sv.values[i], index=feat_names).sort_values(key=abs, ascending=False).head(3)
print(f"\nCustomer #{i} risk={p_te[i]:.2f} reasons:\n", top3.round(3))

# ===== STEP 8: Final evaluation =====
from sklearn.metrics import confusion_matrix
k = int(0.3 * len(p_te)); top = np.argsort(-p_te)[:k]
print("\nTest AUC:", round(roc_auc_score(y_te, p_te), 3),
      "| Recall@top30%:", round(y_te.values[top].sum() / y_te.sum(), 3),
      "| Brier:", round(brier_score_loss(y_te, p_te), 3))
pred_final = predict_fair(p_t, s_te)
print("Confusion matrix [[TN FP][FN TP]]:\n", confusion_matrix(y_te, pred_final))

# ===== STEP 9: Drift (PSI) =====
def psi(expected, actual, bins=10):
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1)); qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected) + 1e-6
    a = np.histogram(actual, qs)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))
print("\nPSI MonthlyCharges (no drift):", round(psi(X_tr["MonthlyCharges"], X_te["MonthlyCharges"]), 3))
print("PSI MonthlyCharges (+10% price hike):", round(psi(X_tr["MonthlyCharges"], X_te["MonthlyCharges"] * 1.10), 3))

# ===== STEP 10: Save the model =====
import joblib
joblib.dump({"preprocessor": pre, "model": model, "group_thresholds": group_t,
             "feature_names": feat_names, "cat_cols": cat_cols}, "churn_model.joblib")
print("\nSaved churn_model.joblib")

# ===== STEP 11: Save plots for the slides =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (a) SHAP top drivers
top6 = imp.head(6)[::-1]
plt.figure(figsize=(7, 4))
plt.barh(top6.index, top6.values, color="#e8643c")
plt.xlabel("Mean |SHAP value|"); plt.title("Top drivers of churn risk")
plt.tight_layout(); plt.savefig("shap_top_drivers.png", dpi=200); plt.close()

# (b) Calibration curve
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(y_te, p_t, n_bins=10)
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
plt.plot(mean_pred, frac_pos, "o-", color="#e8643c", label="XGBoost")
plt.xlabel("Predicted churn probability"); plt.ylabel("Actual churn rate")
plt.title("Calibration"); plt.legend(); plt.tight_layout()
plt.savefig("calibration.png", dpi=200); plt.close()

# (c) Fairness before / after: false-alarm rate by group
fpr_single = MetricFrame(metrics=false_positive_rate, y_true=y_te, y_pred=(p_t >= single_t).astype(int), sensitive_features=s_te).by_group
fpr_fair = MetricFrame(metrics=false_positive_rate, y_true=y_te, y_pred=predict_fair(p_t, s_te), sensitive_features=s_te).by_group
x = np.arange(2)
plt.figure(figsize=(6, 4))
plt.bar(x - 0.2, [fpr_single[0], fpr_single[1]], 0.4, label="Single threshold", color="#bbbbbb")
plt.bar(x + 0.2, [fpr_fair[0], fpr_fair[1]], 0.4, label="Per-group thresholds", color="#e8643c")
plt.xticks(x, ["Non-seniors", "Seniors"]); plt.ylabel("False-alarm rate")
plt.title("Fairness: false alarms by group"); plt.legend(); plt.tight_layout()
plt.savefig("fairness_before_after.png", dpi=200); plt.close()

# (d) Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(y_te, pred_final), display_labels=["Stay", "Churn"]).plot(cmap="Oranges", colorbar=False)
plt.title("Confusion matrix (test set)"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200); plt.close()
print("Saved 4 plots: shap_top_drivers.png, calibration.png, fairness_before_after.png, confusion_matrix.png")
