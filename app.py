# app.py — run with:  python app.py   (after churn_model.py has created churn_model.joblib)
import joblib, numpy as np, pandas as pd, shap, gradio as gr

# Must match the function in churn_model.py (the saved pipeline refers to it by name)
def add_features(d):
    d = d.copy()
    d["AvgMonthlySpend"] = d["TotalCharges"] / d["tenure"].clip(lower=1)
    svc = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
           "TechSupport","StreamingTV","StreamingMovies"]
    d["NumServices"] = (d[svc] == "Yes").sum(axis=1) + (d["InternetService"] != "No")
    d["TotalCharges"] = np.log1p(d["TotalCharges"])
    return d

b = joblib.load("churn_model.joblib")
pre, model, group_t, feat_names, cat_cols = b["preprocessor"], b["model"], b["group_thresholds"], b["feature_names"], b["cat_cols"]
explainer = shap.TreeExplainer(model)

def readable(f, x):
    for c in cat_cols:
        if f.startswith(c + "_"):
            return f"{c} {'=' if x == 1 else 'is not'} {f[len(c)+1:]}"
    return f

def predict(senior, partner, dependents, tenure, phone, multiple, internet, security, backup,
            protection, support, tv, movies, contract, paperless, payment, monthly):
    row = pd.DataFrame([{
        "Partner": partner, "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": monthly * max(tenure, 1)}])
    X = pre.transform(row)
    p = float(model.predict_proba(X)[0, 1])
    g = 1 if senior == "Yes" else 0          # used only to pick the fair threshold
    flagged = p >= group_t[g]

    xs = pd.Series(X[0], index=feat_names)
    sv = pd.Series(explainer(pd.DataFrame(X, columns=feat_names)).values[0], index=feat_names)
    top = sv.sort_values(key=abs, ascending=False).head(3)
    reasons = "\n".join(f"{'▲ raises' if v > 0 else '▼ lowers'} risk ({v:+.2f}): {readable(f, xs[f])}" for f, v in top.items())

    verdict = "⚠️ FLAG for retention offer" if flagged else "✅ No action needed"
    return f"{p:.0%}", verdict, reasons

yn = ["Yes", "No"]
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(yn, value="No", label="Senior citizen"),
        gr.Radio(yn, value="No", label="Partner"),
        gr.Radio(yn, value="No", label="Dependents"),
        gr.Slider(0, 72, value=3, step=1, label="Tenure (months)"),
        gr.Radio(yn, value="Yes", label="Phone service"),
        gr.Radio(["Yes", "No", "No phone service"], value="No", label="Multiple lines"),
        gr.Radio(["DSL", "Fiber optic", "No"], value="Fiber optic", label="Internet service"),
        gr.Radio(["Yes", "No", "No internet service"], value="No", label="Online security"),
        gr.Radio(["Yes", "No", "No internet service"], value="No", label="Online backup"),
        gr.Radio(["Yes", "No", "No internet service"], value="No", label="Device protection"),
        gr.Radio(["Yes", "No", "No internet service"], value="No", label="Tech support"),
        gr.Radio(["Yes", "No", "No internet service"], value="Yes", label="Streaming TV"),
        gr.Radio(["Yes", "No", "No internet service"], value="Yes", label="Streaming movies"),
        gr.Radio(["Month-to-month", "One year", "Two year"], value="Month-to-month", label="Contract"),
        gr.Radio(yn, value="Yes", label="Paperless billing"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                     "Credit card (automatic)"], value="Electronic check", label="Payment method"),
        gr.Number(value=95, label="Monthly charges ($)"),
    ],
    outputs=[gr.Textbox(label="Churn risk"), gr.Textbox(label="Decision"),
             gr.Textbox(label="Top 3 reasons (SHAP)", lines=3)],
    title="Trustworthy Churn Predictor",
    description="Risk score with reasons. The model advises; the retention agent decides.",
)

if __name__ == "__main__":
    demo.launch()
