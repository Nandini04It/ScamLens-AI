# 🛡️ ScamLens AI — UPI Fraud Analytics Dashboard

A Streamlit-based dashboard for exploring UPI fraud-related records, analysing transaction patterns, and providing general fraud-safety guidance.

## 📌 Project Overview

ScamLens AI is an academic data analytics project that explores a dataset of UPI fraud-related records. It uses Python-based data analysis and interactive visualisations to summarise recorded transaction amounts, fraud categories, scam lures, and time-based patterns.

The application also includes an amount-based Decision Tree classifier for categorising predefined fraud-severity labels and a guidance interface for general safety suggestions.

> **Important:** The classifier reproduces predefined severity categories based on transaction amount. It does not independently verify whether a transaction is fraudulent or provide real-time fraud detection.

## 🎯 Objectives

- Explore and summarise UPI fraud-related records.
- Visualise transaction amounts and category distributions.
- Examine patterns by fraud type, scam lure, UPI app, date, and time.
- Demonstrate a Decision Tree classifier for predefined severity categories.
- Provide general fraud-safety guidance through an interactive web interface.

## ✨ Features

- **EDA Dashboard:** Interactive charts and summary statistics.
- **Fraud Analysis:** An interface for entering details and viewing analysis or guidance.
- **Model Insights:** Classification report, confusion matrix, and model evaluation information.
- **Insights & Conclusion:** Key observations, limitations, and possible improvements.
- **Streamlit Interface:** Interactive dashboard for exploring the project.

*Feature availability depends on the files and functionality included in the deployed version.*

## 🧰 Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Plotly and other visualisation libraries used by the application
- Scikit-learn
- Joblib
- CSV datasets

See `requirements.txt` for the project's Python dependencies.

## 📁 Repository Structure

```text
ScamLens-AI/
├── app.py
├── utils.py
├── requirements.txt
├── runtime.txt
├── scam_dataset.csv
├── fraud_severity_model.pkl
├── feature_encoders.pkl
├── target_encoder.pkl
└── README.md
```

The exact files may vary depending on the final application version. Keep all files that `app.py` imports or loads.

## 🚀 Run Locally

1. Install a compatible version of Python.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

5. Start the Streamlit application:

```bash
streamlit run app.py
```

6. Open the local URL displayed in the terminal.

Use the Python version and package versions compatible with the final project configuration.

## 📊 Model and Evaluation

The project includes a Decision Tree classifier for the predefined `Fraud_Severity` target.

The severity labels are derived from transaction amount thresholds:

- **Low:** Amount below ₹5,000
- **Medium:** Amount from ₹5,000 to below ₹20,000
- **High:** Amount of ₹20,000 or above

The model learns amount-based split points to reproduce these labels. A high test accuracy therefore reflects agreement with the predefined labelling rule; it must not be interpreted as proof of real-world fraud-detection accuracy.

## ⚠️ Limitations

- The dataset's real-world provenance has not been independently verified.
- The severity classifier uses transaction amount to reproduce predefined labels.
- The dataset does not establish the probability that a legitimate transaction is fraudulent.
- Income-based impact analysis may use simulated income values; it should not be interpreted as verified victim financial information.
- Safety suggestions are informational and do not guarantee recovery of funds.
- The application is an academic analytics project, not a banking system or real-time fraud-monitoring service.

## 🔮 Future Improvements

- Validate analysis against verified, appropriately anonymised data.
- Evaluate fraud detection using independently verified fraud labels and legitimate transactions.
- Improve validation, privacy protection, and error handling.
- Add more rigorous model evaluation and monitoring if suitable data becomes available.

## 👩‍💻 Project Purpose

Developed as an academic project to demonstrate data analysis, visualisation, basic machine-learning evaluation, and Streamlit application development.

---

**Disclaimer:** This project is for educational and analytical purposes only. It does not replace official banking support, cybercrime reporting, or professional investigation.
