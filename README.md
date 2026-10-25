# 🛡️ ScamLens AI — UPI Fraud Analytics & Recovery Platform

A Streamlit web application that explores UPI fraud records, shows where fraud loss is concentrated, and gives a fraud victim a short report: severity, impact relative to income, a recommended action, and official help contacts.

Built as a project for the *Python with Data Science & ML* internship at **NIELIT Chandigarh** by **Neha** and **Nandini**, under the guidance of Mr. Vineet Kumar.

- 🌐 Live app: https://scamlens-ai.streamlit.app/
- 💻 Repository: https://github.com/Nandini04It/scamlens-ai-

> **Important:** The classifier reproduces predefined severity labels that are derived from the transaction amount. It does not verify whether a transaction is fraudulent and it is not a real-time fraud-detection system.

## 📌 Problem

After a UPI fraud, a victim usually knows only the amount lost. That does not show how serious the case is, how badly it affects this particular person (the same ₹25,000 means different things at different incomes), or what to do next. Organizations such as banks and cyber-crime cells also find it hard to turn many complaints into clear patterns.

ScamLens AI addresses both sides:

- **Analytics** for organizations: which fraud types, lures and amount ranges account for most cases and most loss.
- **Recovery assistant** for victims: severity, income-based financial impact, recommended action, emergency steps and a downloadable PDF report.

## 🎯 Objectives

- Clean and explore a dataset of 1,000 UPI fraud records.
- Engineer features (time slot, day type, severity, loss-to-income ratio, financial impact).
- Visualise patterns by fraud type, lure, UPI app, transaction type and time.
- Demonstrate an explainable Decision Tree for the predefined severity labels.
- Give victims practical next steps and official contacts through a simple web interface.

## ✨ Features

The app has six pages:

| Page | What it offers |
|------|----------------|
| **Home** | Emergency buttons (call 1930, report at cybercrime.gov.in), key statistics, and how the tool works |
| **About** | Project overview, problem statement, dataset, model, workflow and technologies |
| **EDA Dashboard** | Interactive charts in five tabs: Fraud Types and Lures, Apps and Transactions, Time Patterns, Relationships, Financial Impact |
| **Analyze Fraud** | Input form and AI report: severity, loss-vs-income meter, financial impact, recommended action, emergency steps, government help, safety tip, PDF download |
| **Model Insights** | Test accuracy, classification report and confusion matrix |
| **Insights & Conclusion** | Key insights, limitations and future scope |

The details entered in the form are used only to build the report on screen; the app does not save them.

## 🧮 How the analysis works

**Fraud severity** (predicted by the Decision Tree, derived from the amount):

| Severity | Amount lost |
|----------|-------------|
| Low | below ₹5,000 |
| Medium | ₹5,000 to below ₹20,000 |
| High | ₹20,000 or above |

**Financial impact** (simple calculation: amount lost ÷ monthly income × 100):

| Impact | Loss as % of monthly income |
|--------|-----------------------------|
| Low | below 5% |
| Moderate | 5% to below 15% |
| High | 15% to below 40% |
| Critical | 40% or more |

**Recommended action:** the user selects the lure used by the scammer; the app finds the fraud type most often recorded for that lure in the dataset and shows the matching guidance. For High or Critical impact it also shows emergency steps (freeze the account, call the bank, dial 1930, file a complaint at cybercrime.gov.in).

## 📊 Dataset

- File: `scam_dataset.csv` — 1,000 records, 10 columns (`transaction_id`, `date`, `time`, `day_of_week`, `fraud_type`, `fraud_lure`, `amount_inr`, `upi_app`, `transaction_type`, `sender_upi_id`).
- Period covered: April 2023 to March 2025; 10 fraud types, 11 fraud lures, 5 UPI apps, 3 transaction types.
- No missing values and no duplicate rows were found.
- `cleaned_scam_dataset.csv` and `featured_scam_dataset.csv` are the outputs of the cleaning and feature-engineering steps in the notebook.

## 🤖 Model and Evaluation

- **Algorithm:** `DecisionTreeClassifier` (criterion = entropy, max_depth = 5, random_state = 42).
- **Inputs:** `amount_inr`, `Time_Slot`, `Day_Type`, `upi_app`, `transaction_type`, `fraud_lure` (categorical columns label-encoded).
- **Target:** `Fraud_Severity` (Low / Medium / High).
- **Split:** 80% training (800 records), 20% testing (200 records), stratified.
- **Result:** 100% accuracy on the 200 test records.

The trained tree has depth 2 and uses only `amount_inr`, with split points at ₹4,975 and ₹19,975 (equivalent to the ₹5,000 and ₹20,000 limits because all amounts are multiples of ₹50). The model therefore reproduces the labelling rule. The 100% accuracy shows that the pipeline is consistent and explainable; it must not be read as proof of real-world fraud-detection accuracy.

Saved files: `fraud_severity_model.pkl`, `feature_encoders.pkl`, `target_encoder.pkl`.

## 🔍 Key Findings

- Fraud loss is highly skewed: the average loss (₹15,911) is about three times the median (₹5,300), and the largest 5% of cases account for about 41% of the total loss.
- High-severity cases are about 19% of all cases but about 72% of the total loss.
- Phishing and Vishing together make up about 41% of the cases; SIM Swap has the highest average loss.
- Prize/Lottery and KYC-update lures account for the highest total loss.
- Time of day and weekday do not noticeably change the amount lost in this dataset.
- The same loss can have very different impact depending on income, which is why the app reports impact relative to income.

## 🧰 Technologies Used

Python, Streamlit, Pandas, NumPy, Plotly, Scikit-learn, Joblib, fpdf2 (PDF report), Google Colab (analysis notebook). Exact versions are listed in `requirements.txt`.

## 📁 Repository Structure

```text
ScamLens-AI/
├── app.py                      # Streamlit user interface
├── utils.py                    # Backend logic: model loading, prediction, impact, action, PDF report
├── requirements.txt
├── scam_dataset.csv            # Original dataset
├── fraud_severity_model.pkl    # Trained Decision Tree
├── feature_encoders.pkl        # Encoders for input features
├── target_encoder.pkl          # Encoder for the severity label
└── README.md
```

Keep every file that `app.py` and `utils.py` load (the `.pkl` files and `scam_dataset.csv`).

## 🚀 Run Locally

1. Install Python (the project was tested with Python 3.12).
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Start the app:

```bash
streamlit run app.py
```

6. Open the local URL shown in the terminal.

To rerun the analysis, open `ScamLens_AI_NoteBook.ipynb` in Google Colab or Jupyter and run the cells in order. The notebook's "AI Recovery Assistant" section asks for typed input, so run those cells one by one.

> The `.pkl` files must be loaded with a scikit-learn version compatible with the one used to save them. If you see a version warning, install the scikit-learn version recorded in `requirements.txt` or retrain the model by rerunning the notebook.

## ⚠️ Limitations

- The dataset has 1,000 records and represents realistic UPI fraud patterns for academic use; its real-world provenance has not been independently verified, so findings are illustrative and not official fraud statistics.
- The severity classifier reproduces predefined amount-based labels. Apart from the amount, the other attributes (time, day, app, transaction type, lure) showed no meaningful link to severity.
- Monthly income is simulated in the notebook analysis, so impact percentages there are illustrative. In the app, the user enters their own income.
- The dataset contains only fraud cases, so it cannot estimate the probability that a given transaction is fraudulent.
- The recommended action is general guidance matched through the lure and the most common fraud type in the dataset. It does not guarantee recovery of funds.
- This is an academic analytics project, not a banking system or a real-time fraud-monitoring service.

## 🔮 Future Improvements

- Use larger, verified and anonymised complaint data from banks, payment platforms or the cyber-crime portal.
- Collect real (voluntary, anonymous) income bands instead of simulating them.
- Evaluate models on a target that is not defined by a fixed rule, with independently verified fraud labels and legitimate transactions.
- Add a real-time risk score, SMS or e-mail alerts, regional-language support and an investigator dashboard.
- Improve input validation, privacy protection and error handling.

## 🆘 If You Are a Victim of UPI Fraud

- Call the National Cyber Crime Helpline: **1930**
- File a complaint: **https://cybercrime.gov.in**
- Inform your bank immediately and freeze the account or card if needed.

---

**Disclaimer:** This project is for educational and analytical purposes only. It does not replace official banking support, cybercrime reporting, or professional investigation.
