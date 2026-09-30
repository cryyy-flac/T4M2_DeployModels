"""Codificación compartida entre train.py y app.py."""
import pandas as pd

EDU_ORDER = ["High School Diploma", "Associate's Degree", "Bachelor's Degree",
             "Master's Degree", "Doctorate"]
FEATURES = ["Age", "Gender", "Income", "Education", "Marital Status",
            "Number of Children", "Home Ownership"]
# Columnas categóricas (binarias) por posición en FEATURES, para SMOTENC
CAT_IDX = [1, 4, 6]
TARGET = "Credit Score"
LABELS = ["Low", "Average", "High"]  # orden -> 0, 1, 2


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el DataFrame crudo a numérico (mismo orden que FEATURES)."""
    out = pd.DataFrame(index=df.index)
    out["Age"] = df["Age"].astype(float)
    out["Gender"] = (df["Gender"] == "Male").astype(int)
    out["Income"] = df["Income"].astype(float)
    out["Education"] = df["Education"].map({e: i for i, e in enumerate(EDU_ORDER)}).astype(float)
    out["Marital Status"] = (df["Marital Status"] == "Married").astype(int)
    out["Number of Children"] = df["Number of Children"].astype(float)
    out["Home Ownership"] = (df["Home Ownership"] == "Owned").astype(int)
    return out[FEATURES]
