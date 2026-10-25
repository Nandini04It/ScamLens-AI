"""
utils.py

All backend logic for ScamLens AI, kept separate from the Streamlit UI (app.py):

- loading the trained model + encoders
- running a prediction (Fraud Severity)
- financial impact + recommended action + government resources
- PDF report generation

Nothing in this file imports streamlit — it can be unit-tested or reused
on its own.
"""

import joblib
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime


# Load trained artifacts (once, at import time)

model = joblib.load("fraud_severity_model.pkl")
encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# fraud_lure -> most common fraud_type, computed directly from the raw
# dataset (no separate pkl needed — this always stays in sync with
# whatever scam_dataset.csv is sitting next to this file).
_df_raw = pd.read_csv("scam_dataset.csv")
lure_to_type = _df_raw.groupby("fraud_lure")["fraud_type"].agg(lambda x: x.mode()[0]).to_dict()

FEATURES = ["amount_inr", "Time_Slot", "Day_Type", "upi_app", "transaction_type", "fraud_lure"]


# Dropdown options (pulled straight from the encoders, so the UI always
# matches exactly what the model was trained on)

TIME_SLOT_OPTIONS = list(encoders["Time_Slot"].classes_)
DAY_TYPE_OPTIONS = list(encoders["Day_Type"].classes_)
UPI_APP_OPTIONS = list(encoders["upi_app"].classes_)
TRANSACTION_TYPE_OPTIONS = list(encoders["transaction_type"].classes_)
FRAUD_LURE_OPTIONS = list(encoders["fraud_lure"].classes_)


# Rule-based logic (kept identical to the notebook's Feature Engineering
# step, so the app and the notebook always agree)

def financial_impact(ratio: float) -> str:
    """Loss-to-income ratio (%) -> impact category."""
    if ratio < 5:
        return "Low"
    elif ratio < 15:
        return "Moderate"
    elif ratio < 40:
        return "High"
    else:
        return "Critical"


action_map = {
    "Phishing": "Do not click the link again. Block the sender & report the message. File a complaint at cybercrime.gov.in or call 1930.",
    "Vishing": "Never share OTP/PIN over a call. Hang up and call your bank's official number directly. Report to 1930.",
    "QR Code Swap": "Remember: scanning a QR code only SENDS money, never receives it. Verify the merchant before scanning.",
    "Fake Collect Request": "Reject unknown UPI collect requests. A collect request is only used to send money, never to receive it.",
    "OTP Sharing": "Never share your OTP with anyone, including callers claiming to be bank staff. Banks never ask for OTP.",
    "SIM Swap": "If your SIM suddenly stops working, contact your telecom provider immediately and alert your bank. Report to 1930.",
    "Fake UPI App": "Install UPI apps only from the official Play Store / App Store listing. Verify the developer name.",
    "Social Engineering": "Be cautious of urgent or emotional requests for money. Verify the caller's identity independently.",
    "Remote Access Scam": "Never install remote-access apps (AnyDesk, TeamViewer, etc.) on the request of an unknown caller.",
    "Money Mule": "Do not allow your account to be used to transfer money for someone else — this is a criminal offence.",
}
default_action = "Report immediately at cybercrime.gov.in or call the National Cyber Crime Helpline 1930."

# One-line safety tip per fraud type, shown on the report regardless of severity
safety_tips = {
    "Phishing": "Never click links from unknown SMS/WhatsApp/Email messages, even if they look official.",
    "Vishing": "Banks and government bodies never call to ask for your OTP, PIN, or CVV.",
    "QR Code Swap": "You never need to scan a QR code or enter your UPI PIN to *receive* money.",
    "Fake Collect Request": "Always read the request amount and payee name carefully before approving anything.",
    "OTP Sharing": "Your OTP is the same as your signature — never read it out or type it for anyone else.",
    "SIM Swap": "Keep a backup contact number linked to your bank account, not just your primary SIM.",
    "Fake UPI App": "Check the developer name and download count before installing any payment app.",
    "Social Engineering": "Pause before acting on urgency or fear — real emergencies rarely need instant UPI payments.",
    "Remote Access Scam": "No genuine support agent will ever ask you to install AnyDesk or TeamViewer.",
    "Money Mule": "If a stranger asks to route money through your account for a 'commission', it's a crime, not a job.",
}

GOVERNMENT_RESOURCES = [
    {"label": "National Cyber Crime Helpline", "value": "1930", "icon": "☎️", "type": "tel", "link": "tel:1930"},
    {"label": "Cyber Crime Complaint Portal", "value": "cybercrime.gov.in", "icon": "🌐", "type": "url", "link": "https://cybercrime.gov.in"},
    {"label": "NPCI — UPI Safety & Disputes", "value": "npci.org.in", "icon": "🏦", "type": "url", "link": "https://www.npci.org.in"},
    {"label": "RBI — Safe Digital Banking", "value": "rbi.org.in", "icon": "📘", "type": "url", "link": "https://www.rbi.org.in"},
]

EMERGENCY_STEPS = [
    "Freeze your account / card immediately via net-banking or your bank's app",
    "Call your bank's official customer-care helpline (number printed on your card / passbook)",
    "Dial 1930 — National Cyber Crime Helpline",
    "File a complaint at cybercrime.gov.in with transaction screenshots as proof",
]

SEVERITY_COLORS = {"Low": "#22C55E", "Medium": "#F5A623", "High": "#E63946"}
IMPACT_COLORS = {"Low": "#22C55E", "Moderate": "#F5A623", "High": "#F26D3D", "Critical": "#E63946"}



# Main prediction pipeline

def analyze_fraud(amount, monthly_income, time_slot, day_type, upi_app, transaction_type, fraud_lure):
    """
    Runs the full pipeline: ML severity prediction -> financial impact ->
    recommended action -> packages everything the UI needs into one dict.
    """
    sample = pd.DataFrame({
        "amount_inr": [amount],
        "Time_Slot": [time_slot],
        "Day_Type": [day_type],
        "upi_app": [upi_app],
        "transaction_type": [transaction_type],
        "fraud_lure": [fraud_lure],
    })

    for col in sample.columns:
        if col in encoders:
            sample[col] = encoders[col].transform(sample[col])
    sample = sample[FEATURES]

    pred = model.predict(sample)[0]
    severity = target_encoder.inverse_transform([pred])[0]

    loss_ratio = round((amount / monthly_income) * 100, 2) if monthly_income > 0 else 0.0
    impact = financial_impact(loss_ratio)

    fraud_type = lure_to_type.get(fraud_lure)
    recommended_action = action_map.get(fraud_type, default_action)
    safety_tip = safety_tips.get(fraud_type, "Stay alert — never share OTP/PIN, and verify before you pay.")

    return {
        "amount": amount,
        "monthly_income": monthly_income,
        "time_slot": time_slot,
        "day_type": day_type,
        "upi_app": upi_app,
        "transaction_type": transaction_type,
        "fraud_lure": fraud_lure,
        "fraud_type": fraud_type,
        "severity": severity,
        "loss_ratio": loss_ratio,
        "impact": impact,
        "recommended_action": recommended_action,
        "safety_tip": safety_tip,
        "is_emergency": impact in ["High", "Critical"],
    }



# PDF report generation

def _pdf_safe(text: str) -> str:
    """The core Helvetica PDF font only supports latin-1 -- swap out
    characters (em-dash, curly quotes, rupee symbol, etc.) that would
    otherwise crash PDF generation."""
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u20b9": "Rs. ", "\u2026": "...",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def build_pdf_report(result: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 30, 70)
    pdf.cell(0, 12, _pdf_safe("ScamLens AI - Fraud Impact Report"), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, _pdf_safe(f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}"), ln=True)
    pdf.ln(4)

    def section(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 30, 70)
        pdf.cell(0, 10, _pdf_safe(title), ln=True)
        pdf.set_draw_color(200, 200, 210)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

    def row(label, value):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.write(8, _pdf_safe(f"{label} "))
        pdf.set_font("Helvetica", "", 11)
        pdf.write(8, _pdf_safe(str(value)))
        pdf.ln(9)

    section("Transaction Details")
    row("Amount Lost:", f"Rs. {result['amount']:.0f}")
    row("Monthly Income:", f"Rs. {result['monthly_income']:.0f}")
    row("Time Slot:", result["time_slot"])
    row("Day Type:", result["day_type"])
    row("UPI App:", result["upi_app"])
    row("Transaction Type:", result["transaction_type"])
    row("Fraud Lure Used:", result["fraud_lure"])
    row("Likely Fraud Type:", result["fraud_type"] or "Unknown")
    pdf.ln(2)

    section("Risk Summary")
    row("Fraud Severity:", result["severity"])
    row("Loss as % of Income:", f"{result['loss_ratio']}%")
    row("Financial Impact:", result["impact"])
    pdf.ln(2)

    section("Recommended Action")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 7, _pdf_safe(result["recommended_action"]))
    pdf.ln(2)

    if result["is_emergency"]:
        section("Emergency Steps")
        pdf.set_font("Helvetica", "", 11)
        for i, step in enumerate(EMERGENCY_STEPS, start=1):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, _pdf_safe(f"{i}. {step}"))
        pdf.ln(2)

    section("Government Help & Resources")
    pdf.set_font("Helvetica", "", 11)
    for res in GOVERNMENT_RESOURCES:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, _pdf_safe(f"{res['label']}: {res['value']}"))
    pdf.ln(2)

    section("Safety Tip")
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 7, _pdf_safe(result["safety_tip"]))

    return bytes(pdf.output(dest="S"))



# Feature-engineered dataset (for EDA Dashboard)

def load_featured_df() -> pd.DataFrame:
    """Same feature engineering as the notebook's Step 5 — used only for
    EDA charts, kept separate from the ML pipeline above."""
    d = pd.read_csv("scam_dataset.csv")
    d["date"] = pd.to_datetime(d["date"])
    d["time"] = pd.to_datetime(d["time"], format="mixed").dt.time

    def _time_slot(t):
        h = t.hour
        if 5 <= h < 12:
            return "Morning"
        elif 12 <= h < 17:
            return "Afternoon"
        elif 17 <= h < 21:
            return "Evening"
        return "Night"

    d["Time_Slot"] = d["time"].apply(_time_slot)
    d["Day_Name"] = d["date"].dt.day_name()
    d["Day_Type"] = d["Day_Name"].apply(lambda x: "Weekend" if x in ["Saturday", "Sunday"] else "Weekday")
    d["Amount_Category"] = pd.cut(d["amount_inr"], bins=[0, 1000, 10000, float("inf")], labels=["Low", "Medium", "High"])
    d["Fraud_Severity"] = d["amount_inr"].apply(
        lambda a: "Low" if a < 5000 else ("Medium" if a < 20000 else "High")
    )

    # Same simulated income used in the notebook, purely so the EDA
    # dashboard can show a Financial Impact distribution too.
    np.random.seed(42)
    brackets = [12000, 20000, 30000, 45000, 65000, 90000, 130000]
    weights = [0.12, 0.20, 0.24, 0.20, 0.13, 0.08, 0.03]
    income = np.random.choice(brackets, size=len(d), p=weights) + np.random.randint(-2000, 2000, size=len(d))
    d["Monthly_Income"] = np.clip(income, 8000, None)
    d["Loss_to_Income_Ratio"] = ((d["amount_inr"] / d["Monthly_Income"]) * 100).round(2)
    d["Financial_Impact"] = d["Loss_to_Income_Ratio"].apply(financial_impact)

    return d



# Model evaluation metrics (for Model Insights page)

def get_model_metrics():
    """Re-derives train/test evaluation metrics for display — uses the
    exact same split (random_state=42) as training, so results match
    the notebook exactly."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.tree import DecisionTreeClassifier, export_text

    d = load_featured_df()
    X = d[FEATURES].copy()
    y = d["Fraud_Severity"].copy()

    for col in ["Time_Slot", "Day_Type", "upi_app", "transaction_type", "fraud_lure"]:
        X[col] = encoders[col].transform(X[col])
    y_enc = target_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    shallow = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=42)
    shallow.fit(X_train, y_train)
    rules_text = export_text(shallow, feature_names=FEATURES)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classes": list(target_encoder.classes_),
        "report": report,
        "importance": importance,
        "rules_text": rules_text,
        "n_test": len(y_test),
    }
