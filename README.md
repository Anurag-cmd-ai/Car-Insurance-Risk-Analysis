# Car Insurance Claim Risk Intelligence

## Objective

Estimate whether a customer is likely to file a car insurance claim during the policy period, using customer, driving, and vehicle information. The single-screen app presents the model's probability estimate and risk band.

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

## Reproduce the analysis

The numbered Python scripts in `notebooks/` contain the analysis and modeling workflow. Run them from the project root in order:

```bash
python notebooks/01_eda.py
python notebooks/02_baseline.py
python notebooks/03_logistic_regression.py
python notebooks/04_random_forest.py
python notebooks/05_model_comparison.py
python notebooks/06_feature_interpretation.py
python notebooks/07_error_analysis.py
```

The model comparison script saves the selected pipeline to `models/final_pipeline.joblib`. Analysis figures are saved under `reports/figures/`. The original DataCamp notebook is retained in `notebooks/archive/`.

## Run the tests

```bash
python -m pytest tests/ -v
```

## Project layout

```text
app/
  app.py                   Single-screen Streamlit predictor
assets/                    App images and other static assets
dataset/raw/               Original car insurance CSV
models/                    Trained model pipeline
notebooks/                 Numbered analysis and modeling scripts
notebooks/archive/         Archived source notebook
reports/figures/           Generated analysis figures
src/                       Data, preprocessing, training, prediction,
                           evaluation, interpretation, and risk logic
tests/                     Unit tests
requirements.txt            Python dependencies
```

## Author

Anurag Tiwari
[LinkedIn](https://www.linkedin.com/in/anurag-tiwari-7a13743ba/)