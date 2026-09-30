"""Métricas, reportes, matrices de confusión y gráfica comparativa."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from common import LABELS
from .datos import ConjuntoDatos
from .modelos import ModeloBase

IDX = list(range(len(LABELS)))
MACRO = dict(average="macro", zero_division=0)  # cada clase pesa igual (Low tiene muy pocos casos)


class Evaluador:
    def __init__(self, datos: ConjuntoDatos, carpeta="reportes"):
        self.d = datos
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(exist_ok=True)

    def evaluar(self, m: ModeloBase) -> dict:
        p_tr, p_te = m.predecir(self.d.X_train_aug), m.predecir(self.d.X_test)
        acc_tr = accuracy_score(self.d.y_train_aug, p_tr)
        acc_te = accuracy_score(self.d.y_test, p_te)
        return {"Modelo": m.nombre,
                "Accuracy Train": acc_tr, "Accuracy Test": acc_te,
                "Precision Test": precision_score(self.d.y_test, p_te, **MACRO),
                "Recall Test": recall_score(self.d.y_test, p_te, **MACRO),
                "F1-score Test": f1_score(self.d.y_test, p_te, **MACRO),
                "Diferencia Train-Test": acc_tr - acc_te,
                "Accuracy CV": m.cv_score}

    def reporte(self, m: ModeloBase) -> str:
        return classification_report(self.d.y_test, m.predecir(self.d.X_test), labels=IDX,
                                     target_names=LABELS, zero_division=0)

    def guardar_matriz_confusion(self, m: ModeloBase) -> Path:
        cm = confusion_matrix(self.d.y_test, m.predecir(self.d.X_test), labels=IDX)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(IDX, LABELS)
        ax.set_yticks(IDX, LABELS)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(f"Matriz de confusión - {m.nombre}")
        for i in IDX:
            for j in IDX:
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ruta = self.carpeta / f"cm_{m.nombre.replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        return ruta

    def guardar_comparacion(self, tabla) -> Path:
        """Barras agrupadas con las métricas de prueba de todos los modelos."""
        cols = ["Accuracy Test", "Precision Test", "Recall Test", "F1-score Test"]
        x, ancho = np.arange(len(tabla)), 0.2
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, c in enumerate(cols):
            ax.bar(x + (k - 1.5) * ancho, tabla[c], ancho, label=c.replace(" Test", ""))
        ax.set_xticks(x, tabla.index)
        ax.set_ylim(0, 1.05)
        ax.set_title("Comparación de modelos (conjunto de prueba)")
        ax.legend(loc="lower right", fontsize=8)
        ruta = self.carpeta / "comparacion_modelos.png"
        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        return ruta
