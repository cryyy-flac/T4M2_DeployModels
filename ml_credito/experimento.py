"""Orquestación: entrena, evalúa, compara, elige y guarda el mejor modelo."""
import joblib
import pandas as pd

from .datos import ConjuntoDatos
from .evaluacion import Evaluador
from .modelos import ModeloBase


class RepositorioModelo:
    """Persistencia en el formato que espera la app (credit_app.modelo)."""

    def __init__(self, ruta="modelo.joblib"):
        self.ruta = ruta

    def guardar(self, modelo: ModeloBase, rangos: dict) -> None:
        joblib.dump({"model": modelo.pipeline,
                     "meta": {"best": modelo.nombre, "ranges": rangos}}, self.ruta)


class Experimento:
    # Criterio de selección (punto 3 del notebook), en orden de prioridad:
    # métricas macro en prueba (mayor es mejor) y menor brecha |Train-Test| como desempate.
    CRITERIO = {"Accuracy Test": False, "F1-score Test": False, "Precision Test": False,
                "Recall Test": False, "_brecha": True, "Accuracy CV": False}

    def __init__(self, datos: ConjuntoDatos, modelos: list[ModeloBase],
                 evaluador: Evaluador, repositorio: RepositorioModelo):
        self.datos, self.modelos = datos, modelos
        self.evaluador, self.repositorio = evaluador, repositorio

    def ejecutar(self) -> pd.DataFrame:
        d, filas = self.datos, []
        for m in self.modelos:
            print(f"\n=== {m.nombre} ===", flush=True)
            m.ajustar_hiperparametros(d.X_train, d.y_train)
            m.entrenar(d.X_train_aug, d.y_train_aug, d.X_train)
            filas.append(self.evaluador.evaluar(m))
            print(self.evaluador.reporte(m))
            self.evaluador.guardar_matriz_confusion(m)

        tabla = pd.DataFrame(filas).set_index("Modelo").round(4)
        tabla.to_csv(self.evaluador.carpeta / "metricas.csv")
        self.evaluador.guardar_comparacion(tabla)
        print("\n" + tabla.to_markdown())

        tabla_ord = tabla.assign(_brecha=tabla["Diferencia Train-Test"].abs())
        orden = tabla_ord.sort_values(list(self.CRITERIO), ascending=list(self.CRITERIO.values()))
        nombre = orden.index[0]
        mejor = next(m for m in self.modelos if m.nombre == nombre)
        self.repositorio.guardar(mejor, d.rangos)
        print(f"\nMejor modelo: {nombre} → guardado en {self.repositorio.ruta}")
        return tabla
