# 🟣 PROYECTO MAGI — Clasificación de Puntaje Crediticio

```
╔══════════════════════════════════════════════════════════════╗
║  BANCO NACIONAL DE BOB ESPONJA · DIVISIÓN DE ANÁLISIS DE DATOS ║
║  SISTEMA MAGI · MÓDULO 2 · TAREA 4                            ║
║  CLASIFICACIÓN: CONFIDENCIAL                                  ║
╚══════════════════════════════════════════════════════════════╝
```

> *«Los datos son pocos, la hora de la presentación se acerca. Sube al modelo, Ana.»*
> — Comandancia, sala de control

## Resumen de la misión

El Banco Nacional de Bob Esponja necesita un **filtro preliminar** antes de ofrecer tarjetas de crédito o préstamos.
A partir de las características de un cliente (edad, ingresos, estudios, etc.), el sistema predice su **puntaje crediticio**:

| Nivel detectado | Clase (`Credit Score`) | Decisión del banco |
|:---:|:---:|---|
| 🔴 **Bajo** | `Low` | No se otorga ningún producto de crédito |
| 🟡 **Medio** | `Average` | Puede otorgarse un producto con límite medio |
| 🟢 **Alto** | `High` | Puede otorgarse un producto con límite alto |

El proyecto incluye el **entrenamiento de modelos** (programación orientada a objetos) y una **interfaz web** en Gradio para evaluar clientes en tiempo real.

**Dataset:** [Credit Score Classification Dataset](https://www.kaggle.com/datasets/sujithmandala/credit-score-classification-dataset/data) (Kaggle) · 164 registros y 8 variables.

## Pilotos

| Piloto | Rol |
|---|---|
| **Juan** | Piloto de la Unidad Bob Esponja |
| **Oscar** | Piloto de la Unidad Bob Esponja |
| **Ana** | Piloto de la Unidad Bob Esponja |

*(Ajusta aquí el reparto de tareas de cada integrante si lo necesitan.)*

## Estructura del proyecto

```
Tarea4/
├── app.py                  ← lanza la interfaz (Gradio)
├── train.py                ← lanza el entrenamiento
├── common.py               ← codificación y etiquetas compartidas
├── requirements.txt
├── credit_score.csv        ← dataset (colócalo aquí)
├── modelo.joblib           ← se genera al entrenar
│
├── ml_credito/             ← ENTRENAMIENTO (POO)
│   ├── datos.py            ← PreparadorDatos: limpieza, partición y aumento
│   ├── modelos.py          ← ModeloBase (abstracta) + 4 modelos
│   ├── evaluacion.py       ← Evaluador: métricas y gráficas
│   └── experimento.py      ← Experimento: compara, elige y guarda el mejor
│
├── credit_app/             ← INTERFAZ (POO)
│   ├── config.py           ← constantes, rutas y textos
│   ├── modelo.py           ← Cliente, Prediccion, PredictorCrediticio
│   ├── vistas.py           ← Renderizador (plantillas Jinja2)
│   └── interfaz.py         ← InterfazCredito (Gradio)
│
├── templates/              ← HTML de la interfaz
├── assets/                 ← estilos.css y GIF
└── reportes/               ← métricas y gráficas (se genera al entrenar)
```

## Activación del sistema

```bash
# 1. Dependencias
pip install -r requirements.txt

# 2. Entrenar (genera modelo.joblib y la carpeta reportes/)
python train.py credit_score.csv

# 3. Lanzar la interfaz
python app.py
```

Ejecuta los comandos **desde la carpeta raíz** del proyecto para que Python encuentre los módulos.
La interfaz usa Tailwind por CDN, así que necesita **conexión a internet** para verse completa.

## Protocolo de sincronización (flujo de trabajo)

1. **Limpieza:** se detectaron 62 registros duplicados y se eliminaron antes de particionar, para que no aparezcan a la vez en entrenamiento y prueba. Quedan **102 registros únicos**.
2. **Codificación:** `Education` como variable ordinal (0–4) y variables binarias para `Gender`, `Marital Status` y `Home Ownership`.
3. **Partición:** 80 % entrenamiento / 20 % prueba, **estratificada** (la clase `Low` tiene solo 6 casos).
4. **Aumento de datos:** remuestreo con reemplazo ×2 dentro de cada clase, **solo en entrenamiento**. El conjunto de prueba se mantiene intacto.
5. **Escalado:** estandarización de `Age`, `Income` y `Number of Children` para Regresión Logística y KNN.
6. **Modelos y ajuste:** búsqueda de hiperparámetros con `GridSearchCV` (3 folds) sobre el entrenamiento original y entrenamiento final con la muestra aumentada.
7. **Selección:** comparación por accuracy, precision, recall y F1-score (promedio macro) en prueba, más la diferencia Train-Test.
8. **Despliegue:** el mejor modelo se guarda y se consume desde la interfaz.

## Los MAGI (modelos empleados)

| Unidad | Modelo | Escalado |
|:---:|---|:---:|
| MAGI-1 | Regresión Logística | ✅ |
| MAGI-2 | KNN | ✅ |
| MAGI-3 | CART (árbol de decisión) | — |
| MAGI-4 | Random Forest | — |

## Resultados de sincronización

Resultados del análisis del notebook del equipo (accuracy). Al reentrenar con `train.py` pueden variar ligeramente.

| Modelo | Accuracy Train | Accuracy Test |
|---|:---:|:---:|
| Regresión Logística | 98.15 % | 95.24 % |
| KNN | 100 % | 95.24 % |
| **CART** ⭐ | 98.15 % | **100 %** |
| Random Forest | 98.15 % | 95.24 % |

**Modelo seleccionado: CART.**

> ⚠️ **Advertencia de la Comandancia:** el conjunto de prueba tiene solo 21 observaciones y un único cliente `Low`.
> Un 100 % en prueba no garantiza clasificar perfectamente casos nuevos; solo indica que fue el mejor modelo con los datos disponibles.

Las métricas completas y las matrices de confusión quedan en `reportes/` tras entrenar.

## Interfaz

Formulario en español con validaciones para evitar datos absurdos: edad entre 18 y 100, ingreso mayor que 0, hijos enteros. Muestra:

- El nivel de puntaje crediticio y la decisión del banco.
- Las probabilidades por clase.
- Un resumen de los datos capturados.
- Un aviso si los datos caen fuera del rango de entrenamiento.
- Tres clientes de ejemplo para probar rápido.

## Tecnologías

Python · pandas · scikit-learn · matplotlib · Gradio · Jinja2 · Tailwind CSS

---

```
        ── MAGI: RESULTADO APROBADO ──
   CASPER: ✔   BALTHASAR: ✔   MELCHIOR: ✔

      Sincronización completada.
```