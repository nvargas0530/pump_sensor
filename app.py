# -*- coding: utf-8 -*-
"""
Dashboard de Mantenimiento Predictivo — Sensores de Bomba Industrial
====================================================================
Aplicación Dash que materializa los hallazgos del análisis exploratorio
(notebooks/EDA_sensor_pump.ipynb). Permite explorar de forma interactiva:

  * la serie temporal de cualquier sensor con los eventos de falla (BROKEN) marcados,
  * la distribución del sensor seleccionado,
  * su comportamiento según el estado de la máquina (boxplot),
  * indicadores generales del dataset.

Ejecución local:   python app.py   ->   http://127.0.0.1:8050/
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# --------------------------------------------------------------------------- #
# 1. Carga y preparación de datos (mismo tratamiento que el notebook de EDA)
# --------------------------------------------------------------------------- #
DATA_PATH = Path(__file__).parent / "data" / "sensor.csv.gz"

ORDEN_ESTADO = ["NORMAL", "RECOVERING", "BROKEN"]
COLOR_ESTADO = {"NORMAL": "#2ca02c", "RECOVERING": "#ff7f0e", "BROKEN": "#d62728"}


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV comprimido y aplica la limpieza definida en el EDA."""
    df = pd.read_csv(DATA_PATH, compression="infer")

    # Eliminar columnas sin valor analítico
    df = df.drop(columns=[c for c in ["Unnamed: 0", "sensor_15"] if c in df.columns])

    # timestamp como índice temporal
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # machine_status categórica ordenada
    df["machine_status"] = pd.Categorical(
        df["machine_status"], categories=ORDEN_ESTADO, ordered=True
    )

    # Imputación temporal de los sensores numéricos
    sensores = df.select_dtypes(include="number").columns.tolist()
    df[sensores] = df[sensores].interpolate(method="time").ffill().bfill()
    return df


df = cargar_datos()
SENSORES = sorted(df.select_dtypes(include="number").columns.tolist())
FALLAS = df.index[df["machine_status"] == "BROKEN"]

# Métricas de cabecera
N_REGISTROS = len(df)
N_SENSORES = len(SENSORES)
N_FALLAS = int((df["machine_status"] == "BROKEN").sum())
RANGO = f"{df.index.min():%d-%b-%Y} a {df.index.max():%d-%b-%Y}"

# Serie horaria precalculada (acelera el gráfico temporal)
df_hora = df[SENSORES].resample("1h").mean()
estado_hora = (
    df["machine_status"].astype(str).resample("1h").agg(lambda x: x.mode().iat[0] if len(x) else "NORMAL")
)

# --------------------------------------------------------------------------- #
# 2. Inicialización de la app
# --------------------------------------------------------------------------- #
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/flatly/bootstrap.min.css"
]
app = Dash(__name__, external_stylesheets=external_stylesheets,
           title="Pump Sensor Dashboard")
server = app.server  # necesario para despliegue (Render / Railway / Gunicorn)


def tarjeta_kpi(titulo, valor):
    return html.Div(className="kpi-card", children=[
        html.Div(valor, className="kpi-value"),
        html.Div(titulo, className="kpi-label"),
    ])


# --------------------------------------------------------------------------- #
# 3. Layout
# --------------------------------------------------------------------------- #
app.layout = html.Div(className="contenedor", children=[

    html.Div(className="encabezado", children=[
        html.H1("Dashboard de Mantenimiento Predictivo"),
        html.P("Telemetría de 52 sensores de una bomba industrial · "
               "exploración interactiva de señales y eventos de falla"),
    ]),

    # KPIs
    html.Div(className="kpi-row", children=[
        tarjeta_kpi("Registros", f"{N_REGISTROS:,}"),
        tarjeta_kpi("Sensores analizados", f"{N_SENSORES}"),
        tarjeta_kpi("Eventos de falla (BROKEN)", f"{N_FALLAS}"),
        tarjeta_kpi("Periodo", RANGO),
    ]),

    # Controles
    html.Div(className="panel", children=[
        html.Label("Seleccione un sensor:", className="control-label"),
        dcc.Dropdown(
            id="sensor-dropdown",
            options=[{"label": s, "value": s} for s in SENSORES],
            value="sensor_04",
            clearable=False,
        ),
    ]),

    # Serie temporal
    html.Div(className="panel", children=[
        dcc.Graph(id="grafico-serie"),
    ]),

    # Distribución + boxplot lado a lado
    html.Div(className="fila-doble", children=[
        html.Div(className="panel mitad", children=[dcc.Graph(id="grafico-dist")]),
        html.Div(className="panel mitad", children=[dcc.Graph(id="grafico-box")]),
    ]),

    html.Div(className="pie", children=[
        html.P("Fuente: telemetría de bomba (abr–ago 2018). "
               "Proyecto educativo — Módulos 2, 3 y 4."),
    ]),
])


# --------------------------------------------------------------------------- #
# 4. Callbacks
# --------------------------------------------------------------------------- #
@app.callback(
    Output("grafico-serie", "figure"),
    Output("grafico-dist", "figure"),
    Output("grafico-box", "figure"),
    Input("sensor-dropdown", "value"),
)
def actualizar(sensor):
    # --- Serie temporal con eventos de falla -----------------------------
    serie = df_hora[sensor]
    fig_serie = go.Figure()
    fig_serie.add_trace(go.Scatter(
        x=serie.index, y=serie.values, mode="lines",
        name=sensor, line=dict(color="#2c7fb8", width=1)))
    for f in FALLAS:
        fig_serie.add_vline(x=f, line_dash="dash", line_color="#d62728", opacity=0.5)
    fig_serie.update_layout(
        title=f"{sensor} en el tiempo (líneas rojas = fallas BROKEN)",
        xaxis_title="Tiempo", yaxis_title=sensor,
        template="plotly_white", height=420, margin=dict(l=50, r=20, t=60, b=40))

    # --- Distribución ----------------------------------------------------
    fig_dist = px.histogram(
        df, x=sensor, nbins=60, marginal="box",
        title=f"Distribución de {sensor}", template="plotly_white")
    fig_dist.update_layout(height=380, showlegend=False,
                           margin=dict(l=40, r=20, t=60, b=40))

    # --- Boxplot por estado ---------------------------------------------
    fig_box = px.box(
        df.reset_index(), x="machine_status", y=sensor, color="machine_status",
        category_orders={"machine_status": ORDEN_ESTADO},
        color_discrete_map=COLOR_ESTADO,
        title=f"{sensor} según estado de la máquina", template="plotly_white")
    fig_box.update_layout(height=380, showlegend=False,
                          margin=dict(l=40, r=20, t=60, b=40))

    return fig_serie, fig_dist, fig_box


# --------------------------------------------------------------------------- #
# 5. Ejecución
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(debug=True)
