"""Lógica de negocio: cliente, validación y predicción (sin nada de interfaz)."""
from dataclasses import dataclass

import joblib
import pandas as pd

from common import LABELS, encode
from .config import EDUCACION, ESTADO_CIVIL, GENERO, RUTA_MODELO, VIVIENDA


class ErrorValidacion(ValueError):
    """Datos del cliente inválidos."""


@dataclass(frozen=True)
class Cliente:
    edad: int
    genero: str
    ingreso: float
    educacion: str
    estado_civil: str
    hijos: int
    vivienda: str

    def validar(self) -> None:
        if not 18 <= self.edad <= 100:
            raise ErrorValidacion("La edad debe estar entre 18 y 100 años.")
        if self.ingreso <= 0:
            raise ErrorValidacion("El ingreso anual debe ser mayor que 0.")
        if self.hijos < 0 or int(self.hijos) != self.hijos:
            raise ErrorValidacion("El número de hijos debe ser un entero mayor o igual a 0.")

    def a_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Age": self.edad, "Gender": GENERO[self.genero], "Income": self.ingreso,
            "Education": EDUCACION[self.educacion],
            "Marital Status": ESTADO_CIVIL[self.estado_civil],
            "Number of Children": self.hijos, "Home Ownership": VIVIENDA[self.vivienda]}])

    def resumen(self) -> list[tuple[str, str, str]]:
        """(icono, etiqueta, valor) para mostrar chips en el resultado."""
        return [("Edad", f"{self.edad} años"), ("Ingreso", f"${self.ingreso:,.0f}"),
                ("Estudios", self.educacion), ("Estado civil", self.estado_civil),
                ("Hijos", str(int(self.hijos))), ("Vivienda", self.vivienda)]


@dataclass(frozen=True)
class Prediccion:
    nivel: str                       # "Low" | "Average" | "High"
    probabilidades: dict[str, float]
    aviso: str = ""


class PredictorCrediticio:
    def __init__(self, ruta=RUTA_MODELO):
        paquete = joblib.load(ruta)
        self._modelo = paquete["model"]
        rangos = paquete["meta"]["ranges"]
        self.rango_edad, self.rango_ingreso = rangos["Age"], rangos["Income"]

    def _fuera_de_rango(self, c: Cliente) -> bool:
        return not (self.rango_edad[0] <= c.edad <= self.rango_edad[1]
                    and self.rango_ingreso[0] <= c.ingreso <= self.rango_ingreso[1])

    def predecir(self, cliente: Cliente) -> Prediccion:
        cliente.validar()
        proba = self._modelo.predict_proba(encode(cliente.a_dataframe()))[0]
        aviso = ""
        if self._fuera_de_rango(cliente):
            (a0, a1), (i0, i1) = self.rango_edad, self.rango_ingreso
            aviso = (f"Datos fuera del rango de entrenamiento (edad {a0:.0f}–{a1:.0f}, "
                     f"ingreso {i0:,.0f}–{i1:,.0f}). La predicción puede ser poco fiable.")
        return Prediccion(LABELS[int(proba.argmax())],
                          {l: float(p) for l, p in zip(LABELS, proba)}, aviso)
