"""Renderizado de HTML con plantillas Jinja2 (separado de la lógica)."""
import base64

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import NIVELES, RUTA_CSS, RUTA_GIF, RUTA_PLANTILLAS
from .modelo import Cliente, Prediccion


class Renderizador:
    def __init__(self):
        self._env = Environment(loader=FileSystemLoader(RUTA_PLANTILLAS),
                                autoescape=select_autoescape(["html"]))

    def _render(self, plantilla: str, **contexto) -> str:
        return self._env.get_template(plantilla).render(**contexto)

    @staticmethod
    def css() -> str:
        return RUTA_CSS.read_text(encoding="utf-8")

    def cabecera(self) -> str:
        return self._render("cabecera.html")

    def vacio(self) -> str:
        return self._render("vacio.html")

    def gif(self) -> str:
        if not RUTA_GIF.exists():
            return ""
        b64 = base64.b64encode(RUTA_GIF.read_bytes()).decode()
        return self._render("gif.html", b64=b64)

    def resultado(self, cliente: Cliente, pred: Prediccion) -> str:
        barras = [{"nivel": NIVELES[k], "pct": round(p * 100, 1)}
                  for k, p in pred.probabilidades.items()]
        return self._render("resultado.html", nivel=NIVELES[pred.nivel], barras=barras,
                            resumen=cliente.resumen(), aviso=pred.aviso)
