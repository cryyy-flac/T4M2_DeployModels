"""Punto de entrada. Uso: python app.py  (requiere modelo.joblib de train.py)"""
from credit_app.interfaz import InterfazCredito
from credit_app.modelo import PredictorCrediticio
from credit_app.vistas import Renderizador

if __name__ == "__main__":
    InterfazCredito(PredictorCrediticio(), Renderizador()).construir().launch()
