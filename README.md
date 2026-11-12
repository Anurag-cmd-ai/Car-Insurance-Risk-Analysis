# Car Insurance Risk Analytics & Claim Prediction

A robust machine learning system that predicts the probability of a car insurance claim based on customer profile data. Built as an end-to-end project spanning from raw data processing to a deployed Streamlit application.

---

## What This Does

Given a customer profile (including age, driving experience, vehicle type, and credit score), the model calculates an exact claim probability and classifies the customer into **LOW**, **MEDIUM**, or **HIGH** risk tiers. Validated on a held-out test set of 2,000 records, the system achieves:

- **ROC-AUC:** 0.8858
- **F1 Score:** 0.7257
- **Accuracy:** 82.95%

A Logistic Regression model was selected after rigorous benchmarking against Random Forest models across cross-validated ROC-AUC, F1, and calibration metrics.

---

## Applications

1. **Prediction Interface** (`app/streamlit_app.py`) — Enter an individual customer profile to instantly generate claim probabilities and risk verdicts.
2. **Stakeholder Dashboard** (`app/dashboard.py`) — An interactive analytics view featuring claim patterns by segment, feature importance, performance curves, and evidence-backed focus areas.

---

## Project Structure

```text
├── app/
│   ├── streamlit_app.py    # Customer risk prediction interface
│   └── dashboard.py        # Stakeholder analytics dashboard
├── assets/                 # Generated figures and visual media
├── dataset/                # Source data directory
│   └── car_insurance.csv   # Source dataset (10,000 records)
├── models/
│   └── final_pipeline.joblib   # Trained sklearn Pipeline (required for deployment)
├── notebooks/              # Exploratory data analysis and modeling scripts (01–07)
├── reports/                # Evaluation reports and summaries
├── src/                    # Core source modules
│   ├── data.py             # Data loading and schema validation
│   ├── preprocessing.py    # sklearn ColumnTransformer builder
│   ├── train.py            # Train/test split and model persistence
│   ├── evaluate.py         # Metrics, calibration, and error analysis
│   ├── interpret.py        # Feature importance and coefficient extraction
│   ├── predict.py          # Inference layer (loads pipeline, validates input)
│   └── risk.py             # Risk tier classification and focus area detection
├── tests/                  # Pytest unit tests (51 test cases)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Project Python dependencies
└── README.md               # Project documentation
```
---

# Follow these commands in your terminal to set up and run the project locally:
git clone [https://github.com/your-username/Car-Insurance-Risk.git](https://github.com/your-username/Car-Insurance-Risk.git)
cd Car-Insurance-Risk
pip install -r requirements.txt

---

# Author
Anurag Tiwari
https://www.linkedin.com/in/anurag-tiwari-7a13743ba/
