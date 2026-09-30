"""Entrenamiento. Uso: python train.py credit_score.csv"""
import sys

from ml_credito.datos import PreparadorDatos
from ml_credito.evaluacion import Evaluador
from ml_credito.experimento import Experimento, RepositorioModelo
from ml_credito.modelos import CART, KNN, BosqueAleatorio, RegresionLogistica


def main(ruta_csv: str) -> None:
    datos = PreparadorDatos(ruta_csv).preparar()
    modelos = [RegresionLogistica(), KNN(), CART(), BosqueAleatorio()]
    Experimento(datos, modelos, Evaluador(datos), RepositorioModelo()).ejecutar()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "credit_score.csv")
