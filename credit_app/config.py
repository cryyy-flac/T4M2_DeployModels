"""Constantes, rutas y textos de la aplicación."""
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "modelo.joblib"
RUTA_CSS = RAIZ / "assets" / "estilos.css"
RUTA_GIF = RAIZ / "assets" / "spongebob-rainbow-1.gif"
RUTA_PLANTILLAS = RAIZ / "templates"

# Etiqueta en español -> valor que entiende el modelo
GENERO = {"Femenino": "Female", "Masculino": "Male"}
ESTADO_CIVIL = {"Soltero/a": "Single", "Casado/a": "Married"}
VIVIENDA = {"Propia": "Owned", "Rentada": "Rented"}
EDUCACION = {
    "Preparatoria": "High School Diploma",
    "Técnico superior": "Associate's Degree",
    "Licenciatura": "Bachelor's Degree",
    "Maestría": "Master's Degree",
    "Doctorado": "Doctorate",
}

# Tailwind (Play CDN) sin preflight para no romper los estilos de Gradio
HEAD_TAILWIND = (
    '<script src="https://cdn.tailwindcss.com"></script>'
    "<script>tailwind.config={corePlugins:{preflight:false}}</script>"
)


@dataclass(frozen=True)
class Nivel:
    nombre: str
    emoji: str
    color: str
    decision: str


NIVELES = {
    "Low": Nivel("Bajo", "⛔", "#c0392b", "No se otorga ningún producto de crédito."),
    "Average": Nivel("Medio", "⚠️", "#d68910", "Puede otorgarse un producto con límite medio."),
    "High": Nivel("Alto", "✅", "#1e8449", "Puede otorgarse un producto con límite alto."),
}
