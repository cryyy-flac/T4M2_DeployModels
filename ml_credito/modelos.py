"""Modelos de clasificación: una clase base abstracta y una subclase por algoritmo."""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

COLUMNAS_NUMERICAS = ["Age", "Income", "Number of Children"]


class ModeloBase(ABC):
    """Plantilla: cada subclase define su estimador, su rejilla y si requiere escalado."""
    nombre: str = ""
    escalar: bool = False
    escalar_sobre: str = "aumentado"   # datos con los que se ajusta el escalador: "aumentado" | "original"
    param_grid: dict = {}

    def __init__(self, semilla=42, cv=3, n_jobs=-1):
        self.semilla, self.cv, self.n_jobs = semilla, cv, n_jobs
        self.mejores_params: dict = {}
        self.cv_score = float("nan")
        self.pipeline: Pipeline | None = None

    @abstractmethod
    def _estimador(self, **params):
        ...

    def _preprocesador(self):
        if not self.escalar:
            return "passthrough"
        return ColumnTransformer([("esc", StandardScaler(), COLUMNAS_NUMERICAS)],
                                 remainder="passthrough")

    def _armar_pipeline(self, **params) -> Pipeline:
        return Pipeline([("prep", self._preprocesador()), ("modelo", self._estimador(**params))])

    def ajustar_hiperparametros(self, X, y) -> None:
        """GridSearchCV sobre el entrenamiento ORIGINAL (sin aumentar), como en el notebook."""
        if not self.param_grid:
            return
        rejilla = {f"modelo__{k}": v for k, v in self.param_grid.items()}
        busqueda = GridSearchCV(self._armar_pipeline(), rejilla, cv=self.cv,
                                scoring="accuracy", n_jobs=self.n_jobs)
        busqueda.fit(X, y)
        self.mejores_params = {k.removeprefix("modelo__"): v
                               for k, v in busqueda.best_params_.items()}
        self.cv_score = float(busqueda.best_score_)
        print(f"Mejores parámetros: {self.mejores_params} | CV acc: {self.cv_score:.4f}", flush=True)

    def entrenar(self, X_aug, y_aug, X_original) -> "ModeloBase":
        """Ajuste final con la muestra aumentada. Si el modelo escala, el escalador se ajusta
        con `escalar_sobre` (el notebook usa el train original para KNN y el aumentado para LR)."""
        if not self.escalar:
            self.pipeline = self._armar_pipeline(**self.mejores_params).fit(X_aug, y_aug)
            return self
        prep = self._preprocesador().fit(X_original if self.escalar_sobre == "original" else X_aug)
        est = self._estimador(**self.mejores_params).fit(prep.transform(X_aug), y_aug)
        self.pipeline = Pipeline([("prep", prep), ("modelo", est)])
        return self

    def predecir(self, X) -> np.ndarray:
        return self.pipeline.predict(X)


class RegresionLogistica(ModeloBase):
    nombre, escalar = "Regresión Logística", True

    def _estimador(self, **p):
        return LogisticRegression(max_iter=1000, random_state=self.semilla, **p)


class KNN(ModeloBase):
    nombre, escalar, escalar_sobre = "KNN", True, "original"
    param_grid = {"n_neighbors": list(range(1, 16)), "weights": ["uniform", "distance"]}

    def _estimador(self, **p):
        return KNeighborsClassifier(**p)


class CART(ModeloBase):
    nombre = "CART"
    param_grid = {"max_depth": [2, 3, 4, 5, None], "min_samples_split": [2, 4, 6, 8],
                  "min_samples_leaf": [1, 2, 3, 4], "criterion": ["gini", "entropy"]}

    def _estimador(self, **p):
        return DecisionTreeClassifier(random_state=self.semilla, **p)


class BosqueAleatorio(ModeloBase):
    nombre = "Random Forest"
    param_grid = {"n_estimators": [100, 200, 300], "max_depth": [3, 5, None],
                  "min_samples_split": [2, 4, 6], "min_samples_leaf": [1, 2, 3],
                  "max_features": ["sqrt", "log2"]}

    def _estimador(self, **p):
        return RandomForestClassifier(random_state=self.semilla, **p)
