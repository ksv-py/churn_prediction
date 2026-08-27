# Retention Desk

Retention Desk is an original churn analysis and prediction application based on the workflow described in the reference Power BI project. Because the reference repository does not include its documented Excel export, this version uses a relevant public Kaggle telecom churn dataset and clearly records that substitution.

## Data

The source is [Telecom Churn Datasets on Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets). This project uses the real `churn-bigml-80.csv` export, copied locally and renamed to:

`data/telecom_churn.csv`

The export contains state, account length, international plan, voice mail plan, usage minutes and charges, customer service calls, and the `Churn` target.

## What it shows

- Portfolio customer count, churn rate, average tenure, and revenue exposure
- State-level churn concentration
- Plan-level churn pressure
- Churn rate by tenure band with revenue context
- A retention queue ordered by observed revenue for customers labelled `Churned`
- A random forest baseline for churn prioritisation

## Stack

Python, pandas, scikit-learn, Plotly, and Streamlit.

## Run

```powershell
C:\Users\kesha\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements.txt
C:\Users\kesha\AppData\Local\Programs\Python\Python311\python.exe -m streamlit run app.py
```

The app intentionally stops when the Kaggle source file is missing. It does not create synthetic customer records or imply that replacement data represents the reference project.

## Interpretation

The model and charts describe historical association. A churn label is not proof of why a customer left, and revenue at stake is observed revenue, not a forecast. Production use would require leakage checks, time-based validation, calibrated probabilities, cost-sensitive thresholds, fairness review, and a retention experiment.
