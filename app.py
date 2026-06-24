# -*- coding: utf-8 -*-
"""
Dashboard de Mantenimiento Predictivo — Sensores de Bomba Industrial
====================================================================
Versión 2.0 — EDA completo integrado en el dashboard Dash.

Pestañas disponibles:
  1. Visión General        — KPIs + serie temporal interactiva
  2. Calidad de Datos      — nulos, duplicados, tipos de datos
  3. Estadística Descriptiva — media, mediana, std, percentiles, IQR
  4. Distribuciones        — histogramas + boxplots individuales
  5. Outliers              — detección por IQR sensor a sensor
  6. Correlaciones         — heatmap + scatter bivariado
  7. Análisis de Clases    — comportamiento por machine_status

Ejecución local:   python app.py   →   http://127.0.0.1:8050/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# --------------------------------------------------------------------------- #
# 1. Carga y preparación de datos
# --------------------------------------------------------------------------- #
DATA_PATH = Path(__file__).parent / "data" / "sensor.csv.gz"

ORDEN_ESTADO = ["NORMAL", "RECOVERING", "BROKEN"]
COLOR_ESTADO  = {"NORMAL": "#2ca02c", "RECOVERING": "#ff7f0e", "BROKEN": "#d62728"}
PALETTE       = px.colors.qualitative.Set2


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, compression="infer")
    df = df.drop(columns=[c for c in ["Unnamed: 0", "sensor_15"] if c in df.columns])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df["machine_status"] = pd.Categorical(
        df["machine_status"], categories=ORDEN_ESTADO, ordered=True)
    return df


df_raw   = cargar_datos()                                 # sin imputar (para calidad)
df       = df_raw.copy()
sensores = sorted(df.select_dtypes(include="number").columns.tolist())
df[sensores] = df[sensores].interpolate(method="time").ffill().bfill()

FALLAS      = df.index[df["machine_status"] == "BROKEN"]
N_REGISTROS = len(df)
N_SENSORES  = len(sensores)
N_FALLAS    = int((df["machine_status"] == "BROKEN").sum())
RANGO       = f"{df.index.min():%d-%b-%Y}  →  {df.index.max():%d-%b-%Y}"

# Serie horaria para la vista temporal
df_hora      = df[sensores].resample("1h").mean()
estado_hora  = (
    df["machine_status"].astype(str)
    .resample("1h")
    .agg(lambda x: x.mode().iat[0] if len(x) else "NORMAL")
)

# ---- Estadística descriptiva precalculada --------------------------------- #
desc = df[sensores].describe(percentiles=[.25, .5, .75, .95]).T.round(4)
desc["IQR"]     = desc["75%"] - desc["25%"]
desc["missing"] = df_raw[sensores].isnull().sum()
desc["missing%"]= (desc["missing"] / N_REGISTROS * 100).round(2)
desc = desc.rename(columns={"50%": "median"})
desc = desc.reset_index().rename(columns={"index": "sensor"})

# ---- Calidad de datos ----------------------------------------------------- #
nulos_por_col = df_raw.isnull().sum().reset_index()
nulos_por_col.columns = ["columna", "nulos"]
nulos_por_col["porcentaje"] = (nulos_por_col["nulos"] / N_REGISTROS * 100).round(2)
nulos_por_col = nulos_por_col.sort_values("nulos", ascending=False)

tipos_df = df_raw.dtypes.reset_index()
tipos_df.columns = ["columna", "tipo"]
tipos_df["tipo"] = tipos_df["tipo"].astype(str)

# ---- Correlación ---------------------------------------------------------- #
corr_matrix = df[sensores].corr()

# --------------------------------------------------------------------------- #
# 2. App Dash
# --------------------------------------------------------------------------- #
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Pump Sensor EDA Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server


# ─── helpers ────────────────────────────────────────────────────────────────
def kpi_card(titulo, valor, color="#2c3e50"):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.H3(valor, className="card-title text-center fw-bold",
                style={"color": color}),
        html.P(titulo, className="card-text text-center text-muted small"),
    ]), className="shadow-sm border-0 h-100"), md=3, className="mb-3")


SENSOR_OPTS = [{"label": s, "value": s} for s in sensores]

# ─── Layout ─────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[

    # ── Encabezado ──────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.H1("Dashboard de Mantenimiento Predictivo",
                className="text-center text-white mb-1 mt-3",
                style={"fontWeight": 700}),
        html.P("EDA completo · 52 sensores · bomba industrial",
               className="text-center text-white-50 mb-3"),
    ], style={"background": "linear-gradient(135deg,#2c3e50,#3498db)",
              "borderRadius": "10px", "padding": "20px 0"}),
        className="mb-4")),

    # ── KPIs ────────────────────────────────────────────────────────────────
    dbc.Row([
        kpi_card("Registros totales",        f"{N_REGISTROS:,}"),
        kpi_card("Sensores analizados",      f"{N_SENSORES}"),
        kpi_card("Eventos BROKEN",           f"{N_FALLAS}", color="#d62728"),
        kpi_card("Periodo",                  RANGO),
    ], className="mb-2"),

    # ── Tabs ────────────────────────────────────────────────────────────────
    dbc.Tabs(id="tabs", active_tab="tab-overview", children=[

        # ── 1. VISIÓN GENERAL ──────────────────────────────────────────────
        dbc.Tab(label="📈 Visión General", tab_id="tab-overview", children=[
            dbc.Row(dbc.Col([
                html.Label("Sensor:", className="fw-bold mt-3"),
                dcc.Dropdown(id="ov-sensor", options=SENSOR_OPTS,
                             value="sensor_04", clearable=False),
            ], md=4), className="mt-2"),
            dbc.Row(dbc.Col(dcc.Graph(id="ov-serie"), md=12)),
            dbc.Row([
                dbc.Col(dcc.Graph(id="ov-dist"), md=6),
                dbc.Col(dcc.Graph(id="ov-box"),  md=6),
            ]),
        ]),

        # ── 2. CALIDAD DE DATOS ────────────────────────────────────────────
        dbc.Tab(label="🔍 Calidad de Datos", tab_id="tab-quality", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Valores Faltantes por Columna", className="mt-3 fw-bold"),
                    dcc.Graph(id="q-missing-bar"),
                ], md=8),
                dbc.Col([
                    html.H5("Resumen General", className="mt-3 fw-bold"),
                    dbc.ListGroup([
                        dbc.ListGroupItem(f"Registros totales: {N_REGISTROS:,}"),
                        dbc.ListGroupItem(f"Duplicados: 0"),
                        dbc.ListGroupItem(
                            f"Columnas con nulos: "
                            f"{(nulos_por_col['nulos'] > 0).sum()}"),
                        dbc.ListGroupItem(
                            f"sensor_15 — 100 % nulo (eliminado)"),
                        dbc.ListGroupItem(
                            f"sensor_50 — {df_raw['sensor_50'].isnull().mean()*100:.1f} % nulo"),
                    ], flush=True),
                    html.H5("Tipos de Datos", className="mt-4 fw-bold"),
                    dash_table.DataTable(
                        data=tipos_df.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in tipos_df.columns],
                        style_table={"overflowY": "auto", "maxHeight": "300px"},
                        style_cell={"fontSize": 12, "textAlign": "left"},
                        style_header={"fontWeight": "bold",
                                      "background": "#2c3e50", "color": "white"},
                    ),
                ], md=4),
            ]),
            dbc.Row(dbc.Col([
                html.H5("Tabla completa de valores faltantes", className="mt-3 fw-bold"),
                dash_table.DataTable(
                    data=nulos_por_col.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in nulos_por_col.columns],
                    sort_action="native",
                    style_table={"overflowY": "auto", "maxHeight": "350px"},
                    style_cell={"fontSize": 12, "textAlign": "left"},
                    style_header={"fontWeight": "bold",
                                  "background": "#2c3e50", "color": "white"},
                    style_data_conditional=[{
                        "if": {"filter_query": "{porcentaje} > 50"},
                        "backgroundColor": "#fde8e8", "color": "#d62728",
                    }],
                ),
            ])),
        ]),

        # ── 3. ESTADÍSTICA DESCRIPTIVA ─────────────────────────────────────
        dbc.Tab(label="📊 Estadísticas", tab_id="tab-stats", children=[
            dbc.Row([
                dbc.Col([
                    html.H5("Seleccione métrica para comparar sensores",
                            className="mt-3 fw-bold"),
                    dcc.Dropdown(
                        id="st-metric",
                        options=[
                            {"label": "Media",              "value": "mean"},
                            {"label": "Mediana",            "value": "median"},
                            {"label": "Desv. estándar",     "value": "std"},
                            {"label": "IQR",                "value": "IQR"},
                            {"label": "Mínimo",             "value": "min"},
                            {"label": "Máximo",             "value": "max"},
                            {"label": "Percentil 95",       "value": "95%"},
                            {"label": "% Faltantes",        "value": "missing%"},
                        ],
                        value="mean", clearable=False),
                ], md=4),
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id="st-bar"))),
            dbc.Row(dbc.Col([
                html.H5("Tabla resumen completa", className="mt-2 fw-bold"),
                dash_table.DataTable(
                    id="st-table",
                    data=desc.round(4).to_dict("records"),
                    columns=[{"name": c, "id": c}
                             for c in ["sensor","count","mean","median","std",
                                       "min","25%","75%","max","IQR",
                                       "missing","missing%"]],
                    sort_action="native", filter_action="native",
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={"fontSize": 12, "textAlign": "right",
                                "minWidth": "70px"},
                    style_cell_conditional=[
                        {"if": {"column_id": "sensor"},
                         "textAlign": "left", "fontWeight": "bold"}],
                    style_header={"fontWeight": "bold",
                                  "background": "#2c3e50", "color": "white"},
                ),
            ])),
        ]),

        # ── 4. DISTRIBUCIONES ──────────────────────────────────────────────
        dbc.Tab(label="📉 Distribuciones", tab_id="tab-dist", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Sensor:", className="fw-bold mt-3"),
                    dcc.Dropdown(id="di-sensor", options=SENSOR_OPTS,
                                 value="sensor_04", clearable=False),
                ], md=4),
                dbc.Col([
                    html.Label("Nº de bins:", className="fw-bold mt-3"),
                    dcc.Slider(id="di-bins", min=10, max=150, step=5,
                               value=60, marks={i: str(i) for i in [10,40,80,120,150]}),
                ], md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="di-hist"), md=6),
                dbc.Col(dcc.Graph(id="di-violin"), md=6),
            ]),
            dbc.Row(dbc.Col([
                html.H5("Histogramas de todos los sensores (muestra)", className="mt-2 fw-bold"),
                dcc.Graph(id="di-all-hist"),
            ])),
        ]),

        # ── 5. OUTLIERS ────────────────────────────────────────────────────
        dbc.Tab(label="⚠️ Outliers", tab_id="tab-outliers", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Sensor:", className="fw-bold mt-3"),
                    dcc.Dropdown(id="ou-sensor", options=SENSOR_OPTS,
                                 value="sensor_04", clearable=False),
                ], md=4),
                dbc.Col([
                    html.Label("Factor IQR (k):", className="fw-bold mt-3"),
                    dcc.Slider(id="ou-k", min=1.0, max=4.0, step=0.5,
                               value=1.5,
                               marks={v: str(v) for v in [1.0,1.5,2.0,2.5,3.0,4.0]}),
                ], md=5),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="ou-box"), md=6),
                dbc.Col(dcc.Graph(id="ou-scatter"), md=6),
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id="ou-summary-bar"))),
        ]),

        # ── 6. CORRELACIONES ───────────────────────────────────────────────
        dbc.Tab(label="🔗 Correlaciones", tab_id="tab-corr", children=[
            dbc.Row(dbc.Col([
                html.H5("Heatmap de correlación entre sensores",
                        className="mt-3 fw-bold"),
                dcc.Graph(id="co-heatmap"),
            ])),
            dbc.Row([
                dbc.Col([
                    html.Label("Sensor X:", className="fw-bold mt-2"),
                    dcc.Dropdown(id="co-x", options=SENSOR_OPTS,
                                 value="sensor_04", clearable=False),
                ], md=3),
                dbc.Col([
                    html.Label("Sensor Y:", className="fw-bold mt-2"),
                    dcc.Dropdown(id="co-y", options=SENSOR_OPTS,
                                 value="sensor_11", clearable=False),
                ], md=3),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="co-scatter"), md=8),
                dbc.Col([
                    html.H5("Top correlaciones", className="mt-3 fw-bold"),
                    dcc.Graph(id="co-top"),
                ], md=4),
            ]),
        ]),

        # ── 7. ANÁLISIS DE CLASES ──────────────────────────────────────────
        dbc.Tab(label="🏷️ Análisis de Clases", tab_id="tab-class", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Sensor:", className="fw-bold mt-3"),
                    dcc.Dropdown(id="cl-sensor", options=SENSOR_OPTS,
                                 value="sensor_04", clearable=False),
                ], md=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="cl-pie"),   md=4),
                dbc.Col(dcc.Graph(id="cl-box"),   md=8),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="cl-violin"), md=6),
                dbc.Col(dcc.Graph(id="cl-mean-bar"), md=6),
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id="cl-timeline"))),
        ]),

    ]),  # fin Tabs

    html.Hr(),
    html.P("Fuente: telemetría de bomba (abr–ago 2018) · Dashboard EDA v2.0",
           className="text-center text-muted small mb-3"),

])  # fin Container


# =========================================================================== #
# CALLBACKS
# =========================================================================== #

# ── TAB 1: Visión General ──────────────────────────────────────────────────
@app.callback(
    Output("ov-serie", "figure"),
    Output("ov-dist",  "figure"),
    Output("ov-box",   "figure"),
    Input("ov-sensor", "value"),
)
def overview(sensor):
    serie = df_hora[sensor]
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines",
                               name=sensor, line=dict(color="#2c7fb8", width=1)))
    for f in FALLAS:
        fig_s.add_vline(x=f, line_dash="dash", line_color="#d62728", opacity=0.6)
    fig_s.update_layout(
        title=f"{sensor} en el tiempo  (líneas rojas = eventos BROKEN)",
        xaxis_title="Tiempo", yaxis_title=sensor,
        template="plotly_white", height=400, margin=dict(l=50,r=20,t=55,b=40))

    fig_d = px.histogram(df, x=sensor, nbins=60, marginal="box",
                         title=f"Distribución de {sensor}",
                         template="plotly_white", color_discrete_sequence=["#2c7fb8"])
    fig_d.update_layout(height=360, showlegend=False, margin=dict(l=40,r=20,t=55,b=40))

    fig_b = px.box(df.reset_index(), x="machine_status", y=sensor,
                   color="machine_status",
                   category_orders={"machine_status": ORDEN_ESTADO},
                   color_discrete_map=COLOR_ESTADO,
                   title=f"{sensor} por estado de máquina", template="plotly_white")
    fig_b.update_layout(height=360, showlegend=False, margin=dict(l=40,r=20,t=55,b=40))
    return fig_s, fig_d, fig_b


# ── TAB 2: Calidad ────────────────────────────────────────────────────────
@app.callback(Output("q-missing-bar", "figure"), Input("tabs", "active_tab"))
def calidad(_):
    fig = px.bar(
        nulos_por_col[nulos_por_col["nulos"] > 0],
        x="columna", y="porcentaje", text="porcentaje",
        color="porcentaje",
        color_continuous_scale=["#2ca02c","#ff7f0e","#d62728"],
        labels={"porcentaje": "% faltante", "columna": "Columna"},
        title="Porcentaje de valores faltantes por columna",
        template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=400, coloraxis_showscale=False,
                      margin=dict(l=40,r=20,t=55,b=80))
    return fig


# ── TAB 3: Estadísticas ───────────────────────────────────────────────────
@app.callback(Output("st-bar", "figure"), Input("st-metric", "value"))
def stats_bar(metric):
    tmp = desc[["sensor", metric]].sort_values(metric, ascending=False)
    fig = px.bar(tmp, x="sensor", y=metric, text=metric,
                 color=metric, color_continuous_scale="Blues",
                 title=f"{metric} por sensor", template="plotly_white")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False,
                      margin=dict(l=40,r=20,t=55,b=80), xaxis_tickangle=-45)
    return fig


# ── TAB 4: Distribuciones ─────────────────────────────────────────────────
@app.callback(
    Output("di-hist",     "figure"),
    Output("di-violin",   "figure"),
    Output("di-all-hist", "figure"),
    Input("di-sensor", "value"),
    Input("di-bins",   "value"),
)
def distribuciones(sensor, bins):
    # Histograma individual
    fig_h = px.histogram(df, x=sensor, nbins=bins, marginal="rug",
                         color_discrete_sequence=["#3498db"],
                         title=f"Histograma de {sensor}", template="plotly_white")
    fig_h.update_layout(height=380, showlegend=False, margin=dict(l=40,r=20,t=55,b=40))

    # Violin por clase
    fig_v = px.violin(df.reset_index(), y=sensor, x="machine_status",
                      color="machine_status",
                      category_orders={"machine_status": ORDEN_ESTADO},
                      color_discrete_map=COLOR_ESTADO, box=True, points=False,
                      title=f"Violín de {sensor} por clase", template="plotly_white")
    fig_v.update_layout(height=380, showlegend=False, margin=dict(l=40,r=20,t=55,b=40))

    # Grid de histogramas (primeros 16 sensores)
    s16 = sensores[:16]
    rows, cols = 4, 4
    fig_g = make_subplots(rows=rows, cols=cols,
                          subplot_titles=s16, vertical_spacing=0.08)
    for i, s in enumerate(s16):
        r, c = divmod(i, cols)
        vals = df[s].dropna()
        fig_g.add_trace(
            go.Histogram(x=vals, nbinsx=40, showlegend=False,
                         marker_color=PALETTE[i % len(PALETTE)]),
            row=r+1, col=c+1)
    fig_g.update_layout(height=700, title_text="Distribuciones — primeros 16 sensores",
                        template="plotly_white", margin=dict(l=30,r=20,t=70,b=30))
    return fig_h, fig_v, fig_g


# ── TAB 5: Outliers ───────────────────────────────────────────────────────
@app.callback(
    Output("ou-box",         "figure"),
    Output("ou-scatter",     "figure"),
    Output("ou-summary-bar", "figure"),
    Input("ou-sensor", "value"),
    Input("ou-k",      "value"),
)
def outliers(sensor, k):
    Q1, Q3 = df[sensor].quantile(0.25), df[sensor].quantile(0.75)
    IQR_v  = Q3 - Q1
    lo, hi = Q1 - k * IQR_v, Q3 + k * IQR_v
    mask   = (df[sensor] < lo) | (df[sensor] > hi)
    n_out  = int(mask.sum())

    # Boxplot con límites IQR
    fig_b = go.Figure()
    fig_b.add_trace(go.Box(y=df[sensor], name=sensor, boxpoints=False,
                           fillcolor="#2c7fb8", line_color="#1a5276"))
    fig_b.add_hline(y=hi, line_dash="dash", line_color="#d62728",
                    annotation_text=f"Límite sup ({hi:.2f})")
    fig_b.add_hline(y=lo, line_dash="dash", line_color="#d62728",
                    annotation_text=f"Límite inf ({lo:.2f})")
    fig_b.update_layout(title=f"Boxplot {sensor} · k={k}  ({n_out} outliers)",
                        template="plotly_white", height=400,
                        margin=dict(l=50,r=20,t=55,b=40))

    # Scatter temporal con outliers resaltados
    s_hora  = df_hora[sensor]
    colores = ["#d62728" if v < lo or v > hi else "#aec6cf"
               for v in s_hora.values]
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=s_hora.index, y=s_hora.values, mode="markers",
                                marker=dict(color=colores, size=3),
                                name=sensor))
    fig_sc.update_layout(title=f"{sensor} — outliers en rojo (k={k})",
                         template="plotly_white", height=400,
                         margin=dict(l=50,r=20,t=55,b=40))

    # Resumen de outliers por sensor
    n_out_all = {}
    for s in sensores:
        q1, q3 = df[s].quantile(0.25), df[s].quantile(0.75)
        iq = q3 - q1
        n_out_all[s] = int(((df[s] < q1 - k*iq) | (df[s] > q3 + k*iq)).sum())
    out_df = (pd.DataFrame.from_dict(n_out_all, orient="index", columns=["n_outliers"])
              .reset_index().rename(columns={"index": "sensor"})
              .sort_values("n_outliers", ascending=False))
    fig_sum = px.bar(out_df, x="sensor", y="n_outliers",
                     color="n_outliers", color_continuous_scale="Reds",
                     title=f"Cantidad de outliers por sensor (k={k})",
                     template="plotly_white")
    fig_sum.update_layout(height=380, coloraxis_showscale=False,
                          margin=dict(l=40,r=20,t=55,b=80), xaxis_tickangle=-45)
    return fig_b, fig_sc, fig_sum


# ── TAB 6: Correlaciones ──────────────────────────────────────────────────
@app.callback(
    Output("co-heatmap", "figure"),
    Output("co-scatter", "figure"),
    Output("co-top",     "figure"),
    Input("co-x", "value"),
    Input("co-y", "value"),
)
def correlaciones(sx, sy):
    # Heatmap
    fig_h = px.imshow(
        corr_matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
        title="Matriz de correlación (Pearson)", aspect="auto",
        template="plotly_white",
    )
    fig_h.update_layout(height=600, margin=dict(l=20,r=20,t=55,b=20))

    # Scatter bivariado
    sample = df[[sx, sy, "machine_status"]].sample(min(5000, len(df)),
                                                    random_state=42)
    fig_s = px.scatter(
        sample, x=sx, y=sy, color="machine_status",
        color_discrete_map=COLOR_ESTADO, opacity=0.5,
        category_orders={"machine_status": ORDEN_ESTADO},
        title=f"Scatter {sx} vs {sy}  (r = {corr_matrix.loc[sx,sy]:.3f})",
        template="plotly_white",
    )
    # Línea de tendencia manual con numpy (sin statsmodels)
    x_vals = sample[sx].values
    y_vals = sample[sy].values
    mask_valid = ~(np.isnan(x_vals) | np.isnan(y_vals))
    if mask_valid.sum() > 1:
        m, b = np.polyfit(x_vals[mask_valid], y_vals[mask_valid], 1)
        x_line = np.array([x_vals[mask_valid].min(), x_vals[mask_valid].max()])
        y_line = m * x_line + b
        fig_s.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            name="Tendencia", line=dict(color="black", width=2, dash="dash")))
    fig_s.update_layout(height=420, margin=dict(l=40,r=20,t=55,b=40))

    # Top 15 pares más correlacionados
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = (upper.stack()
             .reset_index()
             .rename(columns={"level_0": "s1", "level_1": "s2", 0: "r"})
             .assign(abs_r=lambda x: x["r"].abs())
             .sort_values("abs_r", ascending=False)
             .head(15))
    pairs["par"] = pairs["s1"] + " / " + pairs["s2"]
    fig_t = px.bar(pairs, x="r", y="par", orientation="h",
                   color="r", color_continuous_scale="RdBu_r",
                   range_color=[-1, 1],
                   title="Top 15 pares más correlacionados",
                   template="plotly_white")
    fig_t.update_layout(height=450, coloraxis_showscale=False,
                        margin=dict(l=20,r=20,t=55,b=40))
    return fig_h, fig_s, fig_t


# ── TAB 7: Análisis de Clases ─────────────────────────────────────────────
@app.callback(
    Output("cl-pie",      "figure"),
    Output("cl-box",      "figure"),
    Output("cl-violin",   "figure"),
    Output("cl-mean-bar", "figure"),
    Output("cl-timeline", "figure"),
    Input("cl-sensor", "value"),
)
def clases(sensor):
    df_r = df.reset_index()

    # Pie distribución de clases
    counts = df["machine_status"].value_counts().reset_index()
    counts.columns = ["estado", "n"]
    fig_pie = px.pie(counts, names="estado", values="n",
                     color="estado", color_discrete_map=COLOR_ESTADO,
                     title="Distribución de machine_status", hole=0.4)
    fig_pie.update_layout(height=360, margin=dict(l=20,r=20,t=55,b=20))

    # Boxplot multisensor agrupado por clase
    fig_box = px.box(df_r, x="machine_status", y=sensor,
                     color="machine_status",
                     category_orders={"machine_status": ORDEN_ESTADO},
                     color_discrete_map=COLOR_ESTADO, points=False,
                     title=f"{sensor} por clase", template="plotly_white")
    fig_box.update_layout(height=360, showlegend=False, margin=dict(l=40,r=20,t=55,b=40))

    # Violín
    fig_v = px.violin(df_r, y=sensor, x="machine_status",
                      color="machine_status",
                      category_orders={"machine_status": ORDEN_ESTADO},
                      color_discrete_map=COLOR_ESTADO, box=True, points=False,
                      title=f"Violín {sensor} por clase", template="plotly_white")
    fig_v.update_layout(height=360, showlegend=False, margin=dict(l=40,r=20,t=55,b=40))

    # Medias por clase para todos los sensores (top 20 por diferencia)
    means = (df.groupby("machine_status", observed=False)[sensores]
             .mean()
             .T
             .assign(diff=lambda x: x.max(axis=1) - x.min(axis=1))
             .sort_values("diff", ascending=False)
             .head(20))
    means_long = means.drop(columns="diff").reset_index().melt(
        id_vars="index", var_name="estado", value_name="media")
    means_long.columns = ["sensor", "estado", "media"]
    fig_mb = px.bar(means_long, x="sensor", y="media", color="estado",
                    barmode="group", color_discrete_map=COLOR_ESTADO,
                    category_orders={"estado": ORDEN_ESTADO},
                    title="Media por clase — top 20 sensores más discriminativos",
                    template="plotly_white")
    fig_mb.update_layout(height=420, margin=dict(l=40,r=20,t=55,b=80),
                         xaxis_tickangle=-45)

    # Línea temporal coloreada por estado
    s_hora = df_hora[sensor].reset_index()
    s_hora.columns = ["timestamp", "valor"]
    s_hora["estado"] = estado_hora.values
    fig_tl = px.scatter(s_hora, x="timestamp", y="valor", color="estado",
                        color_discrete_map=COLOR_ESTADO,
                        category_orders={"estado": ORDEN_ESTADO},
                        title=f"{sensor} en el tiempo coloreado por estado",
                        template="plotly_white", opacity=0.7)
    fig_tl.update_traces(marker_size=3)
    fig_tl.update_layout(height=380, margin=dict(l=50,r=20,t=55,b=40))

    return fig_pie, fig_box, fig_v, fig_mb, fig_tl


# =========================================================================== #
if __name__ == "__main__":
    app.run(debug=True)
