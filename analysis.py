from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SOURCE_NAME = "telecom_churn.csv"
SOURCE_URL = "https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets"
TARGET = "Customer_Status"
REQUIRED_COLUMNS = {
    "State", "Account length", "International plan", "Customer service calls", "Churn",
}


def load_churn_data(path: str | Path) -> pd.DataFrame:
    """Load the reference export and add transparent, dashboard-ready fields."""
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Churn export is missing required columns: {', '.join(sorted(missing))}")
    data = data.copy()
    data["Customer_ID"] = [f"TEL-{index:05d}" for index in range(1, len(data) + 1)]
    data["Tenure_in_Months"] = data["Account length"]
    data["Contract"] = data["International plan"].map({"Yes": "International plan", "No": "Domestic plan"})
    data["Monthly_Charge"] = data[["Total day charge", "Total eve charge", "Total night charge", "Total intl charge"]].sum(axis=1)
    data["Total_Revenue"] = data[["Total day charge", "Total eve charge", "Total night charge", "Total intl charge"]].sum(axis=1) * data["Account length"] / 30
    data[TARGET] = data["Churn"].map({True: "Churned", False: "Stayed", "True": "Churned", "False": "Stayed"})
    data["churned"] = data[TARGET].eq("Churned").astype(int)
    data["risk_tier"] = pd.cut(
        data["churned"].astype(float) * 100 + data["Monthly_Charge"].fillna(data["Monthly_Charge"].median()),
        bins=[-float("inf"), 45, 80, float("inf")],
        labels=["Monitor", "Review", "Priority"],
    ).astype(str)
    data["tenure_band"] = pd.cut(
        data["Tenure_in_Months"], bins=[0, 6, 18, 36, float("inf")],
        labels=["0-6 months", "7-18 months", "19-36 months", "37+ months"],
    ).astype(str)
    return data


def train_churn_model(data: pd.DataFrame) -> tuple[Pipeline, float]:
    features = ["Age", "Tenure_in_Months", "Monthly_Charge", "Total_Revenue", "State", "Contract", "Payment_Method"]
    features = [column for column in features if column in data.columns]
    categorical = [column for column in ["State", "Contract", "Payment_Method"] if column in features]
    numeric = [column for column in features if column not in categorical]
    prepared = data[features].copy()
    for column in categorical:
        prepared[column] = prepared[column].fillna("Unknown")
    for column in numeric:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(data[column].median())
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = Pipeline([
        ("features", transformer),
        ("classifier", RandomForestClassifier(n_estimators=180, max_depth=9, class_weight="balanced", random_state=42)),
    ])
    train, test = train_test_split(prepared.assign(target=data["churned"]), test_size=0.25, stratify=data["churned"], random_state=42)
    model.fit(train[features], train["target"])
    score = roc_auc_score(test["target"], model.predict_proba(test[features])[:, 1])
    return model, float(score)


def portfolio_metrics(data: pd.DataFrame) -> dict[str, float]:
    churned = data["churned"].eq(1)
    return {
        "customers": float(len(data)),
        "churn_rate": float(churned.mean()),
        "at_risk_revenue": float(data.loc[churned, "Total_Revenue"].sum()),
        "avg_tenure": float(data["Tenure_in_Months"].mean()),
    }
