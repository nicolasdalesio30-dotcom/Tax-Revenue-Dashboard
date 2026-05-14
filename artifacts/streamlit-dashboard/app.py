"""
Dashboard de Recaudación Tributaria
====================================
Visor interactivo para análisis y proyección de la recaudación tributaria argentina.

Secciones:
    1. Visor del Último Mes    — tortas, KPIs de variación mensual e interanual
    2. Análisis Histórico      — series de tiempo con medias móviles, variaciones
    3. Pronóstico Simple       — Auto-ARIMA sobre la serie histórica del impuesto
    4. Pronóstico Macro        — ARIMA individual para cada variable macroeconómica
    5. Modelo con Macro        — ARIMA con variables exógenas (macro) + comparación
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recaudación Tributaria",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------

# Índice de fila en el DataFrame crudo (0-based, contando desde la fila 1 del Excel)
TAX_ROW_INDEX = {
    "Ganancias":                        12,
    "IVA":                              13,
    "Internos coparticipados":          15,
    "Bienes personales":                21,
    "Créditos y Débitos en cta. cte.":  22,
    "Combustibles Total":               25,
    "Monotributo impositivo":           29,
    "Derechos de importación":          36,
    "Derechos de exportación":          37,
    "Tasa de estadística":              38,
    "Seguridad Social":                 41,
    "Aportes personales":               43,
    "Contribuciones patronales":        44,
    "TOTAL REC. TRIBUTARIOS":           49,
}

# Componentes de la recaudación para el gráfico de torta (excluye totales/subtotales)
COMPONENTES_TORTA = [
    "Ganancias",
    "IVA",
    "Internos coparticipados",
    "Bienes personales",
    "Créditos y Débitos en cta. cte.",
    "Combustibles Total",
    "Monotributo impositivo",
    "Derechos de importación",
    "Derechos de exportación",
    "Tasa de estadística",
    "Aportes personales",
    "Contribuciones patronales",
]

# Paleta de colores para la torta
PALETA_TORTA = px.colors.qualitative.Set3

# Meses en español para parsear el Excel del EMAE
MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

# Colores para las series de pronóstico
COLOR_HISTORICO = "#1f77b4"
COLOR_SIMPLE    = "#ff7f0e"
COLOR_MACRO     = "#2ca02c"

# Ruta al Excel principal de recaudación
EXCEL_PATH = pathlib.Path(__file__).parent.parent.parent / (
    "attached_assets/Recaudación_2023_hasta_2026_Nominal_y_Real_1778714645756.xlsx"
)

# Ruta al Excel de variables macroeconómicas
MACRO_PATH = pathlib.Path(__file__).parent.parent.parent / (
    "attached_assets/Datos_macro_modelo_reca_1778720180502.xlsx"
)

# ---------------------------------------------------------------------------
# CARGA Y PARSEO DE DATOS — RECAUDACIÓN
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando datos de recaudación…")
def cargar_datos() -> dict:
    """
    Lee el Excel de recaudación y devuelve un dict con DataFrames por tipo.
    Cada DataFrame tiene índice DatetimeIndex (mensual) y columnas por impuesto.
    Las variaciones se calculan dinámicamente desde los datos base.
    """
    if not EXCEL_PATH.exists():
        st.error(f"No se encontró el archivo Excel en: {EXCEL_PATH}")
        st.stop()

    raw_nom  = pd.read_excel(EXCEL_PATH, sheet_name="Nominal", header=None)
    raw_real = pd.read_excel(EXCEL_PATH, sheet_name="Real",    header=None)

    def parsear_hoja(raw: pd.DataFrame) -> pd.DataFrame:
        # Fila 9 del Excel (índice 8) tiene las fechas desde columna C (índice 2)
        fechas = pd.to_datetime(raw.iloc[8, 2:], errors="coerce")
        mascara = fechas.notna()
        fechas  = fechas[mascara]
        result  = {}
        for nombre, fila_idx in TAX_ROW_INDEX.items():
            fila   = raw.iloc[fila_idx, 2:][mascara]
            vals   = pd.to_numeric(fila, errors="coerce")
            result[nombre] = pd.Series(vals.values, index=fechas)
        return pd.DataFrame(result)

    df_nom  = parsear_hoja(raw_nom)
    df_real = parsear_hoja(raw_real)

    return {
        "Nominal":      df_nom,
        "Real":         df_real,
        "Var Nominal":  df_nom.pct_change(),
        "Var Real":     df_real.pct_change(),
        "IA Nominal":   df_nom.pct_change(12),
        "IA Real":      df_real.pct_change(12),
    }


# ---------------------------------------------------------------------------
# CARGA Y PARSEO DE DATOS — MACRO
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando datos macroeconómicos…")
def cargar_datos_macro() -> dict:
    """
    Lee el Excel de variables macroeconómicas y devuelve un dict con Series
    mensuales para: IPC, EMAE, Dolar, Tasa.

    - IPC / Dolar / Tasa: datos diarios → último valor del mes
    - EMAE: ya mensual, parseo de año+mes en español
    """
    if not MACRO_PATH.exists():
        st.error(f"No se encontró el archivo macro en: {MACRO_PATH}")
        st.stop()

    result = {}

    # --- IPC: fecha en col A, índice en col B, desde fila 4 ---
    raw_ipc = pd.read_excel(MACRO_PATH, sheet_name="IPC", header=None)
    fechas_ipc = pd.to_datetime(raw_ipc.iloc[3:, 0], errors="coerce")
    vals_ipc   = pd.to_numeric(raw_ipc.iloc[3:, 1], errors="coerce")
    serie_ipc  = pd.Series(vals_ipc.values, index=fechas_ipc).dropna()
    serie_ipc.index = pd.to_datetime(serie_ipc.index).to_period("M").to_timestamp()
    result["IPC"] = serie_ipc

    # --- Dolar oficial: datos diarios, col A=fecha, col C=venta → último del mes ---
    raw_dolar = pd.read_excel(MACRO_PATH, sheet_name="Dolar oficial", header=None)
    fechas_dolar = pd.to_datetime(raw_dolar.iloc[1:, 0], errors="coerce")
    vals_dolar   = pd.to_numeric(raw_dolar.iloc[1:, 2], errors="coerce")
    serie_dolar_d = pd.Series(vals_dolar.values, index=fechas_dolar).dropna()
    # Resamplear al último valor del mes
    serie_dolar = serie_dolar_d.resample("MS").last()
    result["Dolar"] = serie_dolar

    # --- Tasa de interés: datos diarios, col A=fecha, col B=tasa → último del mes ---
    raw_tasa = pd.read_excel(MACRO_PATH, sheet_name="Tasa de interes", header=None)
    fechas_tasa = pd.to_datetime(raw_tasa.iloc[1:, 0], errors="coerce")
    vals_tasa   = pd.to_numeric(raw_tasa.iloc[1:, 1], errors="coerce")
    serie_tasa_d = pd.Series(vals_tasa.values, index=fechas_tasa).dropna()
    serie_tasa   = serie_tasa_d.resample("MS").last()
    result["Tasa"] = serie_tasa

    # --- EMAE: mensual, col A=año (forward-fill), col B=mes en español, col C=índice ---
    raw_emae = pd.read_excel(MACRO_PATH, sheet_name="EMAE", header=None)
    # Datos desde fila 4 (índice 3)
    datos_emae = raw_emae.iloc[3:, :3].copy()
    datos_emae.columns = ["anio", "mes_str", "valor"]
    datos_emae["anio"]    = datos_emae["anio"].ffill()            # llenar año
    datos_emae["anio"]    = pd.to_numeric(datos_emae["anio"],    errors="coerce")
    datos_emae["valor"]   = pd.to_numeric(datos_emae["valor"],   errors="coerce")
    datos_emae["mes_str"] = datos_emae["mes_str"].astype(str).str.strip().str.lower()
    datos_emae["mes_num"] = datos_emae["mes_str"].map(MESES_ES)
    datos_emae = datos_emae.dropna(subset=["anio", "mes_num", "valor"])
    fechas_emae = pd.to_datetime({
        "year":  datos_emae["anio"].astype(int),
        "month": datos_emae["mes_num"].astype(int),
        "day":   1,
    })
    serie_emae = pd.Series(datos_emae["valor"].values, index=fechas_emae)
    result["EMAE"] = serie_emae

    return result


# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES GENERALES
# ---------------------------------------------------------------------------

def filtrar_periodo(df: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Filtra el DataFrame al período seleccionado (6 meses, 1 año, 2 años, Histórico)."""
    if df.empty:
        return df
    ultimo = df.index.max()
    offsets = {"6 meses": 5, "1 año": 11, "2 años": 23}
    if periodo in offsets:
        desde = ultimo - pd.DateOffset(months=offsets[periodo])
        return df[df.index >= desde]
    return df


def formatear_millones(valor: float) -> str:
    """Formatea un número como millones o billones de pesos."""
    if pd.isna(valor):
        return "—"
    if abs(valor) >= 1_000_000:
        return f"${valor/1_000_000:,.1f}B"
    return f"${valor:,.0f}M"


def formatear_porcentaje(valor: float) -> str:
    """Formatea un valor como porcentaje con signo."""
    if pd.isna(valor):
        return "—"
    signo = "+" if valor >= 0 else ""
    return f"{signo}{valor*100:.1f}%"


def calcular_mape(real: np.ndarray, pred: np.ndarray) -> float:
    """Calcula el MAPE (Mean Absolute Percentage Error)."""
    mask = real != 0
    return float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100)


def calcular_rmse(real: np.ndarray, pred: np.ndarray) -> float:
    """Calcula el RMSE (Root Mean Squared Error)."""
    return float(np.sqrt(np.mean((real - pred) ** 2)))


# ---------------------------------------------------------------------------
# COMPONENTES DE GRÁFICOS — RECAUDACIÓN
# ---------------------------------------------------------------------------

def grafico_serie_temporal(serie, nombre_impuesto, tipo_recaudacion, periodo):
    """Línea histórica con medias móviles de 3 y 6 meses."""
    serie_f = filtrar_periodo(serie.to_frame(), periodo)[nombre_impuesto]
    mm3 = filtrar_periodo(serie.rolling(3, min_periods=1).mean().to_frame("mm3"), periodo)["mm3"]
    mm6 = filtrar_periodo(serie.rolling(6, min_periods=1).mean().to_frame("mm6"), periodo)["mm6"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie_f.index, y=serie_f.values, mode="lines+markers",
        name=nombre_impuesto, line=dict(color=COLOR_HISTORICO, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>"))
    fig.add_trace(go.Scatter(x=mm3.index, y=mm3.values, mode="lines",
        name="MM 3 meses", line=dict(color="#ff7f0e", width=2, dash="dot"),
        hovertemplate="<b>MM3 %{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>"))
    fig.add_trace(go.Scatter(x=mm6.index, y=mm6.values, mode="lines",
        name="MM 6 meses", line=dict(color="#2ca02c", width=2, dash="dash"),
        hovertemplate="<b>MM6 %{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>"))
    fig.update_layout(
        title=f"{nombre_impuesto} — {tipo_recaudacion} ({periodo})",
        xaxis_title="Mes", yaxis_title="Millones de pesos",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=450, margin=dict(t=60, b=40, l=60, r=20))
    return fig


def grafico_variacion_barras(data: pd.Series, titulo: str, yaxis_title: str) -> go.Figure:
    """Gráfico de barras con verde/rojo según signo."""
    colores = ["#2ca02c" if v >= 0 else "#d62728" for v in data.values]
    fig = go.Figure(go.Bar(x=data.index, y=data.values * 100, marker_color=colores,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(title=titulo, xaxis_title="Mes", yaxis_title=yaxis_title,
        height=350, margin=dict(t=50, b=40, l=60, r=20))
    return fig


def grafico_torta(df_mes: pd.Series, tipo_recaudacion: str, fecha: pd.Timestamp) -> go.Figure:
    """
    Torta de composición de la recaudación.
    Muestra solo el porcentaje dentro de cada sector; las etiquetas completas
    van en la leyenda para evitar superposiciones con impuestos pequeños.
    """
    valores, etiquetas = [], []
    for comp in COMPONENTES_TORTA:
        v = df_mes.get(comp, np.nan)
        if pd.notna(v) and v > 0:
            valores.append(v)
            etiquetas.append(comp)

    fig = go.Figure(go.Pie(
        labels=etiquetas,
        values=valores,
        hole=0.38,
        marker=dict(colors=PALETA_TORTA),
        # Solo mostramos el % dentro del slice; la etiqueta queda en la leyenda
        textinfo="percent",
        texttemplate="%{percent:.1%}",
        insidetextorientation="radial",
        # Los slices pequeños (<3%) no muestran texto para evitar solapamiento
        textposition="inside",
        automargin=True,
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} M$<br>%{percent:.1%}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Composición de la Recaudación — {tipo_recaudacion} — {fecha.strftime('%B %Y')}",
        height=520,
        margin=dict(t=60, b=20, l=20, r=200),   # margen derecho amplio para la leyenda
        legend=dict(
            orientation="v",
            x=1.02, y=0.5,
            font=dict(size=12),
            title="Impuesto",
        ),
        uniformtext=dict(minsize=10, mode="hide"),  # oculta texto si no cabe
    )
    return fig


def tabla_ultimos_6_meses(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Últimos 6 meses de un impuesto, formateados para mostrar."""
    serie = df[nombre].dropna().tail(6)
    return pd.DataFrame({
        "Mes": [f.strftime("%B %Y") for f in serie.index],
        "Recaudación (M$)": [f"{v:,.0f}" for v in serie.values],
    })


# ---------------------------------------------------------------------------
# SIDEBAR — controles globales
# ---------------------------------------------------------------------------

def render_sidebar(datos: dict) -> tuple:
    """Controles: tipo de recaudación, impuesto y período."""
    st.sidebar.title("⚙️ Controles")
    st.sidebar.markdown("---")

    tipo_recaudacion = st.sidebar.radio(
        "Tipo de recaudación",
        options=["Nominal", "Real"],
        help="Nominal: valores corrientes. Real: ajustados por inflación (base 2023).",
    )

    opciones = list(TAX_ROW_INDEX.keys())
    impuesto = st.sidebar.selectbox(
        "Impuesto",
        options=opciones,
        index=opciones.index("TOTAL REC. TRIBUTARIOS"),
        help="Aplica a todas las pestañas de análisis y pronóstico.",
    )

    st.sidebar.markdown("---")

    periodo = st.sidebar.radio(
        "Período a visualizar",
        options=["6 meses", "1 año", "2 años", "Histórico"],
        index=1,
        help="Filtra el rango de fechas en los gráficos históricos.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Datos: Ministerio de Economía de Argentina. "
        "Valores en millones de pesos corrientes o constantes."
    )

    return tipo_recaudacion, impuesto, periodo


# ---------------------------------------------------------------------------
# SECCIÓN 1: Visor del Último Mes
# ---------------------------------------------------------------------------

def render_ultimo_mes(datos: dict, tipo_recaudacion: str) -> None:
    """KPIs, torta de composición y tabla resumen del último mes disponible."""
    df    = datos[tipo_recaudacion]
    df_vm = datos[f"Var {tipo_recaudacion}"]
    df_ia = datos[f"IA {tipo_recaudacion}"]

    ultimo_mes = df["TOTAL REC. TRIBUTARIOS"].dropna().index.max()
    fila_mes   = df.loc[ultimo_mes]
    total      = fila_mes["TOTAL REC. TRIBUTARIOS"]

    st.subheader(f"📅 Datos del mes: **{ultimo_mes.strftime('%B %Y')}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Recaudación Total", formatear_millones(total))
    col2.metric("Variación mensual",
                v := formatear_porcentaje(df_vm.loc[ultimo_mes, "TOTAL REC. TRIBUTARIOS"]),
                delta=v, delta_color="normal")
    col3.metric("Variación interanual",
                v2 := formatear_porcentaje(df_ia.loc[ultimo_mes, "TOTAL REC. TRIBUTARIOS"]),
                delta=v2, delta_color="normal")

    st.markdown("---")

    # Torta de composición
    fig_torta = grafico_torta(fila_mes, tipo_recaudacion, ultimo_mes)
    st.plotly_chart(fig_torta, width="stretch")

    # Tabla resumen
    st.subheader("Resumen por impuesto")
    rows = []
    for nombre in TAX_ROW_INDEX:
        v   = fila_mes.get(nombre, np.nan)
        vm  = df_vm.loc[ultimo_mes, nombre] if nombre in df_vm.columns else np.nan
        via = df_ia.loc[ultimo_mes, nombre] if nombre in df_ia.columns else np.nan
        rows.append({
            "Impuesto":               nombre,
            "Recaudación (M$)":       f"{v:,.0f}" if pd.notna(v) else "—",
            "Var. mensual":           formatear_porcentaje(vm),
            "Var. interanual (i.a.)": formatear_porcentaje(via),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# SECCIÓN 2: Análisis Histórico
# ---------------------------------------------------------------------------

def render_historico(datos: dict, tipo_recaudacion: str, impuesto: str, periodo: str) -> None:
    """Serie temporal, medias móviles, tabla, variaciones mensual e interanual."""
    df    = datos[tipo_recaudacion]
    df_vm = datos[f"Var {tipo_recaudacion}"]
    df_ia = datos[f"IA {tipo_recaudacion}"]

    st.subheader(f"📈 {impuesto} — {tipo_recaudacion}")

    # Serie temporal + medias móviles
    st.plotly_chart(grafico_serie_temporal(df[impuesto], impuesto, tipo_recaudacion, periodo),
                    width="stretch")

    # Tabla + KPIs de contexto
    col_t, col_k = st.columns([2, 3])
    with col_t:
        st.markdown("**Últimos 6 meses**")
        st.dataframe(tabla_ultimos_6_meses(df, impuesto), width="stretch", hide_index=True)
    with col_k:
        sd = df[impuesto].dropna()
        if len(sd) >= 2:
            ult, ant = sd.iloc[-1], sd.iloc[-2]
            vm  = (ult / ant - 1) if ant else np.nan
            ia  = df_ia[impuesto].dropna()
            iav = ia.iloc[-1] if not ia.empty else np.nan
            st.markdown("**Indicadores del último mes**")
            r1c1, r1c2 = st.columns(2)
            r1c1.metric("Último valor", formatear_millones(ult))
            r1c2.metric("Var. mensual", formatear_porcentaje(vm), delta=formatear_porcentaje(vm))
            r2c1, r2c2 = st.columns(2)
            r2c1.metric("Var. interanual", formatear_porcentaje(iav), delta=formatear_porcentaje(iav))
            r2c2.metric("Máximo histórico", formatear_millones(sd.max()))

    st.markdown("---")

    # Variaciones en dos columnas
    c1, c2 = st.columns(2)
    with c1:
        data = filtrar_periodo(df_vm[impuesto].dropna().to_frame(), periodo)[impuesto]
        st.plotly_chart(grafico_variacion_barras(data, f"Variación mensual — {impuesto}", "Variación (%)"),
                        width="stretch")
    with c2:
        data = filtrar_periodo(df_ia[impuesto].dropna().to_frame(), periodo)[impuesto]
        st.plotly_chart(grafico_variacion_barras(data, f"Variación interanual — {impuesto}", "Variación i.a. (%)"),
                        width="stretch")


# ---------------------------------------------------------------------------
# FUNCIONES DE PRONÓSTICO — helpers internos
# ---------------------------------------------------------------------------

def _ajustar_y_pronosticar(serie: pd.Series, horizonte: int = 6,
                            seasonal: bool = True, m: int = 12,
                            X_hist=None, X_fut=None) -> dict:
    """
    Entrena un modelo auto_arima y genera pronóstico.
    Retorna un dict con: modelo, forecast, conf_int, aic, bic,
    mape_insample, rmse_insample, ljungbox_pvalue.

    Para validación in-sample se dejan los últimos 3 meses como test.
    """
    from pmdarima import auto_arima
    from statsmodels.stats.diagnostic import acorr_ljungbox

    n_test = 3  # meses reservados para validación

    # --- Validación in-sample (train/test split) ---
    y_train_val = serie.iloc[:-n_test].values
    y_test_val  = serie.iloc[-n_test:].values

    try:
        modelo_val = auto_arima(
            y_train_val,
            X=X_hist[:-n_test] if X_hist is not None else None,
            seasonal=seasonal, m=m,
            stepwise=True, suppress_warnings=True,
            error_action="ignore", max_order=10,
        )
        pred_val = modelo_val.predict(
            n_periods=n_test,
            X=X_hist[-n_test:] if X_hist is not None else None,
        )
        mape_val = calcular_mape(y_test_val, pred_val)
        rmse_val = calcular_rmse(y_test_val, pred_val)
    except Exception:
        mape_val, rmse_val = np.nan, np.nan

    # --- Modelo final sobre toda la serie ---
    modelo_final = auto_arima(
        serie.values,
        X=X_hist,
        seasonal=seasonal, m=m,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", max_order=10,
    )

    forecast, conf_int = modelo_final.predict(
        n_periods=horizonte,
        X=X_fut,
        return_conf_int=True,
        alpha=0.10,  # intervalo de confianza del 90%
    )

    # Test de Ljung-Box sobre residuos (lags=10)
    try:
        residuos = modelo_final.resid()
        lb = acorr_ljungbox(residuos, lags=[10], return_df=True)
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_pvalue = np.nan

    return {
        "modelo":       modelo_final,
        "forecast":     forecast,
        "conf_int":     conf_int,
        "aic":          modelo_final.aic(),
        "bic":          modelo_final.bic(),
        "mape":         mape_val,
        "rmse":         rmse_val,
        "lb_pvalue":    lb_pvalue,
        "orden":        str(modelo_final.order),
    }


@st.cache_data(show_spinner=False)
def pronosticar_serie_cache(vals: tuple, fechas_iso: tuple,
                             horizonte: int = 6,
                             seasonal: bool = True, m: int = 12) -> dict:
    """
    Wrapper cacheable para pronóstico simple (sin exógenas).
    Los args son tuplas para que Streamlit pueda hashearlos.
    """
    serie = pd.Series(list(vals), index=pd.to_datetime(list(fechas_iso)))
    try:
        res = _ajustar_y_pronosticar(serie, horizonte=horizonte, seasonal=seasonal, m=m)
        # Convertir a tipos serializables para caché
        return {
            "forecast":  res["forecast"].tolist(),
            "ci_lower":  res["conf_int"][:, 0].tolist(),
            "ci_upper":  res["conf_int"][:, 1].tolist(),
            "aic":       res["aic"],
            "bic":       res["bic"],
            "mape":      res["mape"],
            "rmse":      res["rmse"],
            "lb_pvalue": res["lb_pvalue"],
            "orden":     res["orden"],
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def pronosticar_macro_cache(vals: tuple, fechas_iso: tuple,
                             horizonte: int = 6) -> dict:
    """
    Wrapper cacheable para pronóstico de una variable macro (sin exógenas, sin estacionalidad).
    """
    serie = pd.Series(list(vals), index=pd.to_datetime(list(fechas_iso)))
    try:
        res = _ajustar_y_pronosticar(serie, horizonte=horizonte, seasonal=False)
        return {
            "forecast":   res["forecast"].tolist(),
            "ci_lower":   res["conf_int"][:, 0].tolist(),
            "ci_upper":   res["conf_int"][:, 1].tolist(),
            "aic":        res["aic"],
            "bic":        res["bic"],
            "orden":      res["orden"],
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def pronosticar_con_exogenas_cache(
    y_vals: tuple, y_fechas: tuple,
    X_hist_dict: dict, X_fut_dict: dict,
    horizonte: int = 6,
) -> dict:
    """
    Wrapper cacheable para pronóstico ARIMA con variables exógenas.

    X_hist_dict y X_fut_dict son dicts {nombre_macro: lista_valores}.
    """
    serie = pd.Series(list(y_vals), index=pd.to_datetime(list(y_fechas)))

    # Construir matrices de exógenas (alineadas a la serie y)
    nombres_macro = list(X_hist_dict.keys())
    X_hist = np.column_stack([X_hist_dict[k] for k in nombres_macro])
    X_fut  = np.column_stack([X_fut_dict[k]  for k in nombres_macro])

    try:
        res = _ajustar_y_pronosticar(
            serie, horizonte=horizonte,
            seasonal=True, m=12,
            X_hist=X_hist, X_fut=X_fut,
        )
        return {
            "forecast":  res["forecast"].tolist(),
            "ci_lower":  res["conf_int"][:, 0].tolist(),
            "ci_upper":  res["conf_int"][:, 1].tolist(),
            "aic":       res["aic"],
            "bic":       res["bic"],
            "mape":      res["mape"],
            "rmse":      res["rmse"],
            "lb_pvalue": res["lb_pvalue"],
            "orden":     res["orden"],
        }
    except Exception as e:
        return {"error": str(e)}


def generar_fechas_futuras(ultima_fecha: pd.Timestamp, horizonte: int) -> pd.DatetimeIndex:
    """Genera las próximas `horizonte` fechas mensuales después de `ultima_fecha`."""
    return pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=1),
        periods=horizonte,
        freq="MS",
    )


def grafico_pronostico(
    serie_hist: pd.Series,
    fechas_fut: pd.DatetimeIndex,
    forecast: list,
    ci_lower: list,
    ci_upper: list,
    titulo: str,
    nombre_serie: str = "Histórico",
    nombre_forecast: str = "Pronóstico",
    color_hist: str = COLOR_HISTORICO,
    color_fc: str = COLOR_SIMPLE,
    show_hist_n: int = 24,   # cuántos meses históricos mostrar
) -> go.Figure:
    """
    Gráfico de línea con histórico reciente + pronóstico + banda de confianza.
    """
    hist_plot = serie_hist.dropna().tail(show_hist_n)
    fc_arr  = np.array(forecast)
    ci_lo   = np.array(ci_lower)
    ci_hi   = np.array(ci_upper)

    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(
        x=hist_plot.index, y=hist_plot.values,
        mode="lines+markers", name=nombre_serie,
        line=dict(color=color_hist, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.1f}<extra></extra>",
    ))

    # Banda de confianza (90%)
    fig.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=list(ci_hi) + list(ci_lo[::-1]),
        fill="toself", fillcolor=f"rgba({int(color_fc[1:3], 16)},{int(color_fc[3:5], 16)},{int(color_fc[5:], 16)},0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 90%", showlegend=True,
        hoverinfo="skip",
    ))

    # Pronóstico
    fig.add_trace(go.Scatter(
        x=fechas_fut, y=fc_arr,
        mode="lines+markers", name=nombre_forecast,
        line=dict(color=color_fc, width=2.5, dash="dash"), marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.1f}<extra></extra>",
    ))

    # Línea vertical separando histórico de pronóstico
    fig.add_vline(x=str(serie_hist.dropna().index.max()), line_dash="dot",
                  line_color="gray", line_width=1)

    fig.update_layout(
        title=titulo,
        xaxis_title="Mes", yaxis_title="Valor",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=420,
        margin=dict(t=70, b=40, l=60, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# SECCIÓN 3: Pronóstico Simple (Auto-ARIMA sin exógenas)
# ---------------------------------------------------------------------------

def render_pronostico_simple(datos: dict, tipo_recaudacion: str, impuesto: str) -> None:
    """
    Pestaña 'Pronóstico simple':
      - Entrena auto_arima(seasonal=True, m=12) sobre la serie histórica del impuesto.
      - Muestra histórico + pronóstico a 6 meses con banda de confianza.
      - Métricas: AIC, BIC, MAPE, RMSE, Ljung-Box p-value.
    """
    st.subheader(f"🔮 Pronóstico Simple — {impuesto} ({tipo_recaudacion})")
    st.caption(
        "Modelo **Auto-ARIMA** (estacional, m=12) entrenado únicamente con la serie "
        "histórica de recaudación. La validación usa los últimos 3 meses como test."
    )

    df = datos[tipo_recaudacion]
    serie = df[impuesto].dropna()

    if len(serie) < 18:
        st.warning("La serie tiene menos de 18 observaciones. El pronóstico puede ser poco fiable.")

    horizonte = 6

    with st.spinner("⏳ Entrenando Auto-ARIMA (puede tardar unos segundos)…"):
        res = pronosticar_serie_cache(
            vals=tuple(serie.values.tolist()),
            fechas_iso=tuple(serie.index.strftime("%Y-%m-%d").tolist()),
            horizonte=horizonte,
            seasonal=True, m=12,
        )

    if "error" in res:
        st.error(f"Error al entrenar el modelo: {res['error']}")
        return

    fechas_fut = generar_fechas_futuras(serie.index.max(), horizonte)

    # --- Gráfico ---
    fig = grafico_pronostico(
        serie_hist=serie,
        fechas_fut=fechas_fut,
        forecast=res["forecast"],
        ci_lower=res["ci_lower"],
        ci_upper=res["ci_upper"],
        titulo=f"{impuesto} — Pronóstico simple a {horizonte} meses",
        nombre_serie="Histórico",
        nombre_forecast="Pronóstico ARIMA",
    )
    st.plotly_chart(fig, width="stretch")

    # --- Tabla de valores pronosticados ---
    df_fc = pd.DataFrame({
        "Mes":              [f.strftime("%B %Y") for f in fechas_fut],
        "Pronóstico (M$)":  [f"{v:,.0f}" for v in res["forecast"]],
        "IC Inferior":      [f"{v:,.0f}" for v in res["ci_lower"]],
        "IC Superior":      [f"{v:,.0f}" for v in res["ci_upper"]],
    })
    st.markdown("**Valores pronosticados (con intervalo de confianza del 90%)**")
    st.dataframe(df_fc, hide_index=True)

    st.markdown("---")

    # --- Métricas del modelo ---
    st.markdown("**Métricas del modelo**")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)

    mc1.metric("AIC",   f"{res['aic']:.1f}"  if pd.notna(res["aic"])   else "—")
    mc2.metric("BIC",   f"{res['bic']:.1f}"  if pd.notna(res["bic"])   else "—")
    mc3.metric("MAPE",  f"{res['mape']:.1f}%" if pd.notna(res["mape"]) else "—",
               help="Error porcentual medio absoluto — validación en los últimos 3 meses.")
    mc4.metric("RMSE",  f"{res['rmse']:,.0f}" if pd.notna(res["rmse"]) else "—",
               help="Raíz del error cuadrático medio — validación en los últimos 3 meses.")
    mc5.metric("Ljung-Box p-value", f"{res['lb_pvalue']:.3f}" if pd.notna(res["lb_pvalue"]) else "—",
               help="p-value del test de Ljung-Box (lags=10). Valores > 0.05 indican residuos sin autocorrelación significativa.")

    with st.expander("ℹ️ Interpretación de métricas"):
        st.markdown("""
| Métrica | Descripción |
|---|---|
| **AIC** | Criterio de información de Akaike. Menor es mejor (para comparar modelos). |
| **BIC** | Criterio de información bayesiano. Similar al AIC, penaliza más la complejidad. |
| **MAPE** | Error porcentual medio absoluto. Indica cuánto se desvía el pronóstico en % (calculado con los últimos 3 meses reales como test). |
| **RMSE** | Raíz del error cuadrático medio. En las mismas unidades que la variable (M$). |
| **Ljung-Box** | Prueba si los residuos son ruido blanco. p > 0.05 indica buen ajuste. |
        """)
        st.markdown(f"**Orden del modelo seleccionado:** `ARIMA{res['orden']}`")


# ---------------------------------------------------------------------------
# SECCIÓN 4: Pronóstico Macro (ARIMA individual por variable)
# ---------------------------------------------------------------------------

def render_pronostico_macro() -> dict:
    """
    Pestaña 'Pronóstico Macro':
      - Carga las 4 variables macroeconómicas (IPC, EMAE, Dolar, Tasa).
      - Entrena un auto_arima individual para cada una.
      - Muestra 4 subgráficos con histórico + pronóstico a 6 meses.
      - Devuelve los pronósticos futuros para uso en la Etapa 3.
    """
    st.subheader("📉 Pronóstico de Variables Macroeconómicas")
    st.caption(
        "Se entrena un modelo **Auto-ARIMA** independiente para cada variable macro. "
        "Los valores proyectados se utilizarán como insumo del modelo con exógenas (Etapa 3)."
    )

    macro = cargar_datos_macro()

    horizonte = 6

    # Configuración de cada variable macro: nombre, unidad, color
    config_macro = {
        "IPC":   dict(label="IPC (Índice de Precios al Consumidor)", unidad="Índice",  color="#e377c2"),
        "EMAE":  dict(label="EMAE (Índice de Actividad Económica)",   unidad="Índice",  color="#8c564b"),
        "Dolar": dict(label="Tipo de Cambio — Dólar Oficial (Venta)", unidad="$/USD",   color="#bcbd22"),
        "Tasa":  dict(label="Tasa de Interés (depósitos 30d)",        unidad="TNA (%)", color="#17becf"),
    }

    pronosticos_futuros = {}   # se devuelven para la Etapa 3
    resultados_macro    = {}   # para mostrar gráficos

    # Entrenar modelos con spinner
    with st.spinner("⏳ Entrenando modelos ARIMA para variables macro…"):
        for key, cfg in config_macro.items():
            serie = macro.get(key)
            if serie is None or serie.empty:
                resultados_macro[key] = {"error": "Sin datos"}
                continue
            serie = serie.dropna()
            res = pronosticar_macro_cache(
                vals=tuple(serie.values.tolist()),
                fechas_iso=tuple(serie.index.strftime("%Y-%m-%d").tolist()),
                horizonte=horizonte,
            )
            resultados_macro[key] = res
            if "error" not in res:
                fechas_fut = generar_fechas_futuras(serie.index.max(), horizonte)
                pronosticos_futuros[key] = pd.Series(res["forecast"], index=fechas_fut)

    # --- Mostrar gráficos en grilla 2×2 ---
    keys = list(config_macro.keys())
    for fila_idx in range(0, len(keys), 2):
        cols = st.columns(2)
        for col_idx, key in enumerate(keys[fila_idx: fila_idx + 2]):
            with cols[col_idx]:
                cfg = config_macro[key]
                res = resultados_macro.get(key, {})
                serie = macro.get(key, pd.Series(dtype=float)).dropna()

                if "error" in res:
                    st.error(f"{cfg['label']}: {res['error']}")
                    continue

                fechas_fut = generar_fechas_futuras(serie.index.max(), horizonte)

                fig = grafico_pronostico(
                    serie_hist=serie,
                    fechas_fut=fechas_fut,
                    forecast=res["forecast"],
                    ci_lower=res["ci_lower"],
                    ci_upper=res["ci_upper"],
                    titulo=cfg["label"],
                    nombre_serie="Histórico",
                    nombre_forecast="Pronóstico",
                    color_hist=COLOR_HISTORICO,
                    color_fc=cfg["color"],
                    show_hist_n=30,
                )
                fig.update_layout(yaxis_title=cfg["unidad"], height=380)
                st.plotly_chart(fig, width="stretch")

                # Métricas compactas debajo del gráfico
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("AIC", f"{res['aic']:.1f}" if pd.notna(res["aic"]) else "—")
                mc2.metric("BIC", f"{res['bic']:.1f}" if pd.notna(res["bic"]) else "—")
                mc3.metric("Orden", res.get("orden", "—"))

    st.markdown("---")
    st.success(
        "✅ Los pronósticos macro están listos. "
        "Andá a la pestaña **Modelo con Macro** para ver cómo mejoran el pronóstico de recaudación."
    )

    return pronosticos_futuros


# ---------------------------------------------------------------------------
# SECCIÓN 5: Modelo con Macro (ARIMA con variables exógenas)
# ---------------------------------------------------------------------------

def render_modelo_con_macro(datos: dict, tipo_recaudacion: str, impuesto: str) -> None:
    """
    Pestaña 'Modelo con Macro':
      - Alinea la serie de recaudación con el histórico de las 4 macros.
      - Entrena auto_arima con X (exógenas) usando valores históricos reales.
      - Pronóstica 6 meses usando los valores proyectados de las macros (Etapa 2).
      - Compara el MAPE del modelo simple vs. el modelo con macro.
    """
    st.subheader("🧠 Modelo con Variables Macroeconómicas (Exógenas)")
    st.info(
        "Las proyecciones de las variables macro se obtienen de modelos ARIMA individuales. "
        "El modelo con exógenas incorpora estos pronósticos para mejorar la predicción "
        "de la recaudación.",
        icon="ℹ️",
    )

    df   = datos[tipo_recaudacion]
    y    = df[impuesto].dropna()
    macro = cargar_datos_macro()
    horizonte = 6

    # --- Alinear fechas: intersección entre y y cada macro ---
    macro_disponibles = {k: v.dropna() for k, v in macro.items() if v is not None and not v.empty}

    # Encontrar el rango de fechas común
    fecha_inicio = max(y.index.min(), *[s.index.min() for s in macro_disponibles.values()])
    fecha_fin    = min(y.index.max(), *[s.index.max() for s in macro_disponibles.values()])

    y_alineada = y[(y.index >= fecha_inicio) & (y.index <= fecha_fin)]

    # Construir matrices X históricas (alineadas a y)
    X_hist_dict = {}
    for key, serie in macro_disponibles.items():
        # Reindexar a las fechas de y_alineada con forward-fill para posibles gaps
        s_alin = serie.reindex(y_alineada.index, method="ffill")
        if s_alin.isna().sum() > len(s_alin) * 0.5:
            st.warning(f"La macro {key} tiene demasiados valores faltantes en el rango alineado. Se omite.")
            continue
        s_alin = s_alin.fillna(method="ffill").fillna(method="bfill")
        X_hist_dict[key] = s_alin.values.tolist()

    if not X_hist_dict:
        st.error("No hay variables macro disponibles para alinear con la serie de recaudación.")
        return

    # --- Pronósticos futuros de las macros (Etapa 2, recomputados si es necesario) ---
    X_fut_dict = {}
    with st.spinner("⏳ Computando pronósticos macro para el período futuro…"):
        for key, serie in macro_disponibles.items():
            if key not in X_hist_dict:
                continue
            res_m = pronosticar_macro_cache(
                vals=tuple(serie.values.tolist()),
                fechas_iso=tuple(serie.index.strftime("%Y-%m-%d").tolist()),
                horizonte=horizonte,
            )
            if "error" not in res_m:
                X_fut_dict[key] = res_m["forecast"]

    if not X_fut_dict:
        st.error("No se pudieron obtener pronósticos para las variables macro.")
        return

    # Asegurar mismas macros en hist y fut
    macros_comunes = [k for k in X_hist_dict if k in X_fut_dict]
    X_hist_dict = {k: X_hist_dict[k] for k in macros_comunes}
    X_fut_dict  = {k: X_fut_dict[k]  for k in macros_comunes}

    # --- Entrenar modelo simple (para comparar) ---
    with st.spinner("⏳ Entrenando modelo simple (referencia)…"):
        res_simple = pronosticar_serie_cache(
            vals=tuple(y_alineada.values.tolist()),
            fechas_iso=tuple(y_alineada.index.strftime("%Y-%m-%d").tolist()),
            horizonte=horizonte, seasonal=True, m=12,
        )

    # --- Entrenar modelo con exógenas ---
    with st.spinner("⏳ Entrenando modelo ARIMA con variables macroeconómicas…"):
        res_macro = pronosticar_con_exogenas_cache(
            y_vals=tuple(y_alineada.values.tolist()),
            y_fechas=tuple(y_alineada.index.strftime("%Y-%m-%d").tolist()),
            X_hist_dict={k: v for k, v in X_hist_dict.items()},
            X_fut_dict={k: v for k, v in X_fut_dict.items()},
            horizonte=horizonte,
        )

    if "error" in res_macro:
        st.error(f"Error en modelo con exógenas: {res_macro['error']}")
        return
    if "error" in res_simple:
        st.error(f"Error en modelo simple: {res_simple['error']}")
        return

    fechas_fut = generar_fechas_futuras(y_alineada.index.max(), horizonte)

    # --- Gráfico comparativo: histórico + ambos pronósticos ---
    fig = go.Figure()

    # Histórico (últimos 24 meses)
    hist_plot = y_alineada.tail(24)
    fig.add_trace(go.Scatter(
        x=hist_plot.index, y=hist_plot.values,
        mode="lines+markers", name="Histórico",
        line=dict(color=COLOR_HISTORICO, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>",
    ))

    # Banda IC del modelo simple
    fig.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=res_simple["ci_upper"] + res_simple["ci_lower"][::-1],
        fill="toself", fillcolor="rgba(255,127,14,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="IC Simple", hoverinfo="skip",
    ))

    # Pronóstico simple
    fig.add_trace(go.Scatter(
        x=fechas_fut, y=res_simple["forecast"],
        mode="lines+markers", name="Pronóstico Simple (ARIMA)",
        line=dict(color=COLOR_SIMPLE, width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>",
    ))

    # Banda IC del modelo con macro
    fig.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=res_macro["ci_upper"] + res_macro["ci_lower"][::-1],
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="IC con Macro", hoverinfo="skip",
    ))

    # Pronóstico con macro
    fig.add_trace(go.Scatter(
        x=fechas_fut, y=res_macro["forecast"],
        mode="lines+markers", name="Pronóstico con Macro (ARIMAX)",
        line=dict(color=COLOR_MACRO, width=2.5, dash="dot"),
        marker=dict(size=7, symbol="circle"),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>",
    ))

    fig.add_vline(x=str(y_alineada.index.max()), line_dash="dot",
                  line_color="gray", line_width=1)

    fig.update_layout(
        title=f"{impuesto} — Comparación: Simple vs. con Macro (exógenas: {', '.join(macros_comunes)})",
        xaxis_title="Mes", yaxis_title="Millones de pesos",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=500,
        margin=dict(t=80, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width="stretch")

    # --- Tabla comparativa de pronósticos ---
    st.markdown("**Valores pronosticados — comparación mes a mes**")
    df_comp = pd.DataFrame({
        "Mes":                      [f.strftime("%B %Y") for f in fechas_fut],
        "Simple (M$)":              [f"{v:,.0f}" for v in res_simple["forecast"]],
        "Con Macro (M$)":           [f"{v:,.0f}" for v in res_macro["forecast"]],
        "Diferencia (M$)":          [f"{(m-s):+,.0f}"
                                     for s, m in zip(res_simple["forecast"], res_macro["forecast"])],
    })
    st.dataframe(df_comp, hide_index=True)

    st.markdown("---")

    # --- Comparación de métricas MAPE/RMSE ---
    st.markdown("**Comparación de métricas (validación en últimos 3 meses)**")

    mape_s = res_simple.get("mape", np.nan)
    mape_m = res_macro.get("mape",  np.nan)
    rmse_s = res_simple.get("rmse", np.nan)
    rmse_m = res_macro.get("rmse",  np.nan)
    lb_s   = res_simple.get("lb_pvalue", np.nan)
    lb_m   = res_macro.get("lb_pvalue",  np.nan)

    # Mejora porcentual del MAPE
    if pd.notna(mape_s) and pd.notna(mape_m) and mape_s != 0:
        mejora_mape = (mape_s - mape_m) / mape_s * 100
        mejora_str  = f"{mejora_mape:+.1f}%"
        mejora_help = "Positivo = el modelo con macro tiene menor error"
    else:
        mejora_str, mejora_help = "—", ""

    mc = st.columns(5)
    mc[0].metric("MAPE Simple",    f"{mape_s:.1f}%" if pd.notna(mape_s) else "—")
    mc[1].metric("MAPE con Macro", f"{mape_m:.1f}%" if pd.notna(mape_m) else "—")
    mc[2].metric("Mejora MAPE",    mejora_str, help=mejora_help)
    mc[3].metric("RMSE Simple",    f"{rmse_s:,.0f}" if pd.notna(rmse_s) else "—")
    mc[4].metric("RMSE con Macro", f"{rmse_m:,.0f}" if pd.notna(rmse_m) else "—")

    # Ljung-Box
    lb_col1, lb_col2, _ = st.columns(3)
    lb_col1.metric("Ljung-Box p (Simple)", f"{lb_s:.3f}" if pd.notna(lb_s) else "—",
                   help="p > 0.05: residuos sin autocorrelación significativa")
    lb_col2.metric("Ljung-Box p (Macro)",  f"{lb_m:.3f}" if pd.notna(lb_m) else "—",
                   help="p > 0.05: residuos sin autocorrelación significativa")

    with st.expander("ℹ️ Sobre las variables exógenas utilizadas"):
        st.markdown(f"""
Las siguientes variables macro fueron incluidas como exógenas en el modelo ARIMAX:

| Variable | Descripción |
|---|---|
| **IPC** | Índice de Precios al Consumidor (base dic 2016). Ajusta por inflación. |
| **EMAE** | Estimador Mensual de Actividad Económica. Proxy del nivel de actividad. |
| **Dólar** | Tipo de cambio oficial (venta). Impacta en derechos de exportación/importación. |
| **Tasa** | Tasa de interés de depósitos a 30 días. Indicador de política monetaria. |

Los valores futuros de cada macro se proyectan con modelos ARIMA individuales (Etapa 2).
        """)
        st.markdown(
            f"**Macros usadas:** {', '.join(macros_comunes)} | "
            f"**Período de entrenamiento:** {y_alineada.index.min().strftime('%b %Y')} → "
            f"{y_alineada.index.max().strftime('%b %Y')} ({len(y_alineada)} obs.)"
        )


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

def main() -> None:
    """Función principal — orquesta el dashboard con todas sus secciones."""

    st.title("📊 Dashboard de Recaudación Tributaria")
    st.caption(
        "Análisis y proyección de la recaudación tributaria argentina. "
        "Fuente: datos internos de la oficina. Valores en millones de pesos."
    )

    datos = cargar_datos()
    tipo_recaudacion, impuesto, periodo = render_sidebar(datos)

    # Cinco pestañas principales
    tab_mes, tab_hist, tab_simple, tab_macro, tab_exog = st.tabs([
        "📅 Visor del Último Mes",
        "📈 Análisis Histórico",
        "🔮 Pronóstico Simple",
        "📉 Pronóstico Macro",
        "🧠 Modelo con Macro",
    ])

    with tab_mes:
        render_ultimo_mes(datos, tipo_recaudacion)

    with tab_hist:
        render_historico(datos, tipo_recaudacion, impuesto, periodo)

    with tab_simple:
        render_pronostico_simple(datos, tipo_recaudacion, impuesto)

    with tab_macro:
        render_pronostico_macro()

    with tab_exog:
        render_modelo_con_macro(datos, tipo_recaudacion, impuesto)


if __name__ == "__main__":
    main()
