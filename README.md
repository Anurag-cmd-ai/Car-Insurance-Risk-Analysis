## Car Insurance Claim Risk Intelligence
A robust machine learning system that predicts the probability of a car insurance claim based on customer profile data. Built as an end-to-end project from raw data through to a deployed Streamlit application.

# Live Link -
Hosted on Streamlit Community Cloud.
https://car-insurance-risk-analysis-5b.streamlit.app/

<img width="1407" height="876" alt="Screenshot 2026-11-12 202838" src="https://github.com/user-attachments/assets/a0cbb1a2-6753-4ed2-8476-96348b97dcd6" />


## Objective

Estimate whether a customer is likely to file a car insurance claim during the policy period, using customer, driving, and vehicle information. The single-screen app presents the model's probability estimate and risk band.

## Scope
Given a customer profile (age, driving experience, vehicle type, credit score, etc.), the model outputs a claim probability and classifies the customer as LOW, MEDIUM, or HIGH risk. The system was validated on a held-out test set of 2,000 records and achieves:

ROC-AUC: 0.8858
F1 Score: 0.7257
Accuracy: 82.95%

## How it helps insurers

Insurers can use the estimate as one input when reviewing modeled claim likelihood and potential exposure during underwriting and pricing. The app provides decision support; it does not set premiums or approve or reject coverage.

## Tech stack

- **Python** for the application and analysis scripts
- **pandas** and **NumPy** for data handling
- **scikit-learn** for preprocessing, model training, and prediction
- **Streamlit** for the predictor interface
- **Matplotlib**, **Seaborn**, and **Plotly** for analysis visuals
- **joblib** for saving and loading the trained model
- **statsmodels** for the archived notebook reference
- **pytest** for the test suite

## Project layout
```
car-insurance-risk/
├── app/
│   └── app.py                       # Single-screen Streamlit predictor
├── assets/
│   └── car.jpg
├── data/
│   ├── raw/
│   │   └── car_insurance.csv
│   └── processed/                   # Generated data; usually Git-ignored
├── models/
│   └── final_pipeline.joblib
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_baseline.py
│   ├── 03_logistic_regression.py
│   ├── 04_random_forest.py
│   ├── 05_model_comparison.py
│   ├── 06_feature_interpretation.py
│   ├── 07_error_analysis.py
│   └── archive/
├── reports/
│   └── figures/
├── src/
│   └── car_insurance_risk/
│       ├── __init__.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── predict.py
│       ├── risk.py
│       ├── evaluate.py
│       └── interpret.py
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## Model results

The selected Logistic Regression pipeline achieved the following results on the held-out test set:

- **ROC-AUC:** 0.8858
- **F1 score:** 0.7257
- **Accuracy:** 82.95%

It was selected after comparison with a Random Forest model using cross-validated ROC-AUC, F1, and calibration metrics.

## Run the app

From the project root, install the dependencies and start the predictor:

```bash
python -m pip install -r requirements.txt
streamlit run app/app.py
```

Enter a customer and vehicle profile and select **Assess claim risk**. The app shows the predicted claim probability, a binary claim-risk flag using a 50% cutoff, and the model's LOW, MEDIUM, or HIGH risk band.

The app loads the trained pipeline from `models/final_pipeline.joblib`.

## Author

Anurag Tiwari
[LinkedIn](https://www.linkedin.com/in/anurag-tiwari-7a13743ba/)
