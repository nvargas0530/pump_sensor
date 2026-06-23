# Mantenimiento Predictivo de Bomba Industrial — EDA + Dashboard con Dash

Proyecto de ciencia de datos que analiza la telemetría de **52 sensores** de una bomba
industrial (muestreo de 1 minuto, abril–agosto 2018) para **entender y anticipar las
fallas** de la máquina. Incluye un **análisis exploratorio reproducible** (Jupyter) y un
**dashboard interactivo** desarrollado con **Dash** y **Plotly**.

Aplica de forma integral los **Módulos 2 (EDA), 3 (tratamiento de datos y dataviz
interactivo), 4 (despliegue de dashboard) y 5 (machine learning)** del curso *Ciencia de
datos con Python para la toma de decisiones*.

## Problema de Negocio

Una parada no planificada de la bomba detiene la producción y genera sobrecostos. Las 52
señales disponibles se registran cada minuto, pero **no toda la telemetría es útil**:
algunos sensores están vacíos, otros son redundantes y las fallas reales son
extremadamente raras. El reto es transformar datos crudos y ruidosos en información
accionable que permita **detectar anomalías que preceden a una falla**.

## Impacto del Negocio

- **Mitigación de riesgos:** identificación de las desviaciones de telemetría que
  acompañan a los 7 eventos de falla (`BROKEN`) del periodo.
- **Reducción de redundancia:** detección de 65 pares de sensores fuertemente
  correlacionados (|r| > 0,9), base para reducir dimensionalidad antes de modelar.
- **Eficiencia operativa:** un dashboard interactivo que reemplaza la inspección manual de
  decenas de señales por una exploración guiada.

---

## Estructura del proyecto

```
pump_sensor_dash/
├── app.py                       # Dashboard interactivo (Dash)
├── requirements.txt             # Dependencias
├── Dockerfile                   # Imagen para contenedor
├── Procfile                     # Despliegue (Render / Railway / Heroku)
├── assets/
│   └── style.css                # Estilos del dashboard
├── data/
│   ├── sensor.csv.gz            # Dataset crudo comprimido (gzip)
│   └── preprocess.py            # Script de limpieza -> sensor_clean.csv.gz
└── notebooks/
    └── EDA_ML_sensor_pump.ipynb # EDA + Machine Learning (Módulos 2, 3 y 5)
```

> El dataset se distribuye comprimido (`.csv.gz`, ~37 MB) porque el CSV original (~119 MB)
> supera el límite de 100 MB por archivo de GitHub. `pandas` lo lee directamente.

---

## Paso a Paso para ejecutar la aplicación

### 0. Requisitos previos

Tener instalado [Conda](https://docs.anaconda.com/miniconda/) (o `venv`).

#### 0.1 Crear el entorno virtual

    conda create -n pump_env python=3.10 -y

#### 0.2 Activar el entorno

    conda activate pump_env

### 1. Clonar el repositorio

    git clone https://github.com/<TU_USUARIO>/pump_sensor_dash.git
    cd pump_sensor_dash

### 2. Instalar dependencias

    pip install -r requirements.txt

### 3. (Opcional) Generar el dataset limpio

    python data/preprocess.py

### 4. Explorar el análisis (EDA + Machine Learning)

    jupyter notebook notebooks/EDA_ML_sensor_pump.ipynb

### 5. Iniciar el dashboard

    python app.py

### 6. Acceder al dashboard

Abra su navegador en: **http://127.0.0.1:8050/**

---

## Principales hallazgos del EDA

| Tema | Resultado |
|------|-----------|
| Registros | 220.320 (1 abr – 31 ago 2018, cada minuto) |
| Calidad | `sensor_15` 100 % nulo (eliminado); resto imputado por interpolación temporal |
| Duplicados | 0 filas y 0 marcas de tiempo repetidas |
| Distribuciones | No normales (asimetría y curtosis altas; Shapiro–Wilk rechaza normalidad) |
| Redundancia | 65 pares de sensores con \|r\| > 0,9 |
| Variable objetivo | Desbalance extremo: solo 7 eventos `BROKEN` |
| **Mejor modelo (ML)** | **Random Forest** — F1 ≈ 0,997, ROC-AUC ≈ 1,0 (detección de estado anómalo) |

---

## Despliegue en producción

El proyecto incluye `Procfile` y `Dockerfile`. Para Render/Railway basta con conectar el
repositorio y usar el comando `gunicorn app:server`. La variable `server = app.server` en
`app.py` expone el servidor Flask subyacente requerido por estas plataformas.

## Notas técnicas

- Stack: Python 3.10, Dash, Plotly, Pandas, NumPy, SciPy, Seaborn, Matplotlib.
- El tratamiento de datos del dashboard es idéntico al del notebook, garantizando
  coherencia entre el análisis y la herramienta interactiva.
