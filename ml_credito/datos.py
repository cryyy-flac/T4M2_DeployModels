"""Carga, limpieza, partición y aumento de datos."""
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from common import LABELS, TARGET, encode


@dataclass(frozen=True)
class ConjuntoDatos:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_aug: pd.DataFrame
    y_train_aug: pd.Series
    rangos: dict  # rangos de Age/Income vistos, para avisar al usuario en la app


class PreparadorDatos:
    """Replica el procesamiento del notebook: sin duplicados, codificación,
    partición estratificada y remuestreo con reemplazo SOLO en entrenamiento."""

    def __init__(self, ruta_csv, test_size=0.20, factor_aumento=2, semilla=42):
        self.ruta_csv = ruta_csv
        self.test_size = test_size
        self.factor = factor_aumento
        self.semilla = semilla

    def cargar(self) -> pd.DataFrame:
        df = pd.read_csv(self.ruta_csv)
        print(f"Registros originales: {len(df)} | duplicados: {df.duplicated().sum()}")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Registros únicos: {len(df)}")
        return df

    def _aumentar(self, X: pd.DataFrame, y: pd.Series):
        datos = X.copy()
        datos[TARGET] = y.values
        aug = pd.concat([
            grupo.sample(n=len(grupo) * self.factor, replace=True, random_state=self.semilla)
            for _, grupo in datos.groupby(TARGET)
        ]).reset_index(drop=True)
        return aug.drop(columns=[TARGET]), aug[TARGET]

    def preparar(self) -> ConjuntoDatos:
        df = self.cargar()
        X = encode(df)
        y = df[TARGET].str.strip().map({l: i for i, l in enumerate(LABELS)}).astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=self.semilla, stratify=y)
        X_aug, y_aug = self._aumentar(X_tr, y_tr)
        print(f"Train: {len(X_tr)} → aumentado: {len(X_aug)} | Test: {len(X_te)}")
        rangos = {c: (float(X[c].min()), float(X[c].max())) for c in ("Age", "Income")}
        return ConjuntoDatos(X_tr, X_te, y_tr, y_te, X_aug, y_aug, rangos)
