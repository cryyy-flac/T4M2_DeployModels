"""Construcción de la interfaz Gradio."""
import gradio as gr

from .config import EDUCACION, ESTADO_CIVIL, GENERO, HEAD_TAILWIND, VIVIENDA
from .modelo import Cliente, ErrorValidacion, PredictorCrediticio
from .vistas import Renderizador

VALORES_INICIALES = (30, "Femenino", 50000, "Licenciatura", "Soltero/a", 0, "Rentada")
EJEMPLOS = [
    [24, "Masculino", 28000, "Preparatoria", "Soltero/a", 0, "Rentada"],
    [38, "Femenino", 70000, "Licenciatura", "Casado/a", 2, "Propia"],
    [45, "Masculino", 140000, "Doctorado", "Casado/a", 1, "Propia"],
]


class InterfazCredito:
    def __init__(self, predictor: PredictorCrediticio, vistas: Renderizador):
        self.predictor = predictor
        self.vistas = vistas

    # ---- callbacks ----
    def _evaluar(self, edad, genero, ingreso, educacion, civil, hijos, vivienda) -> str:
        if None in (edad, ingreso, hijos):
            raise gr.Error("Completa todos los campos numéricos.")
        cliente = Cliente(int(edad), genero, float(ingreso), educacion, civil, hijos, vivienda)
        try:
            prediccion = self.predictor.predecir(cliente)
        except ErrorValidacion as e:
            raise gr.Error(str(e))
        return self.vistas.resultado(cliente, prediccion)

    def _limpiar(self):
        return (*VALORES_INICIALES, self.vistas.vacio())

    # ---- construcción ----
    def construir(self) -> gr.Blocks:
        tema = gr.themes.Base(primary_hue="amber", secondary_hue="sky", neutral_hue="stone")
        with gr.Blocks(title="Banco Nacional de Bob Esponja", theme=tema,
                       css=self.vistas.css(), head=HEAD_TAILWIND) as demo:
            gr.HTML(self.vistas.cabecera())
            with gr.Row():
                with gr.Column(elem_classes="panel"):
                    gr.Markdown("### Datos del cliente")
                    edad = gr.Slider(18, 100, step=1, value=VALORES_INICIALES[0], label="Edad (años)")
                    genero = gr.Radio(list(GENERO), value=VALORES_INICIALES[1], label="Género")
                    ingreso = gr.Number(value=VALORES_INICIALES[2], precision=0, minimum=0, label="Ingreso anual (USD)")
                    educacion = gr.Dropdown(list(EDUCACION), value=VALORES_INICIALES[3], label="Nivel educativo",
                                            filterable=False)
                    civil = gr.Radio(list(ESTADO_CIVIL), value=VALORES_INICIALES[4], label="Estado civil")
                    hijos = gr.Slider(0, 10, step=1, value=VALORES_INICIALES[5], label="Número de hijos")
                    vivienda = gr.Radio(list(VIVIENDA), value=VALORES_INICIALES[6], label="Vivienda")
                    with gr.Row():
                        btn = gr.Button("Evaluar cliente", elem_id="btn-evaluar")
                        limpiar = gr.Button("Limpiar", elem_id="btn-limpiar")
                with gr.Column(elem_classes="panel"):
                    gr.Markdown("### Resultado")
                    salida = gr.HTML(self.vistas.vacio())
                    gr.HTML(self.vistas.gif())

            entradas = [edad, genero, ingreso, educacion, civil, hijos, vivienda]
            gr.Examples(EJEMPLOS, entradas, label="Clientes de ejemplo")
            btn.click(self._evaluar, entradas, salida)
            limpiar.click(self._limpiar, None, entradas + [salida])
        return demo
