"""
Dashboard de Recaudación Tributaria
====================================
Visor interactivo para análisis de recaudación tributaria argentina.
Fuente de datos: Excel con pestañas Nominal, Real, Variación Nominal, Variación Real.

Secciones:
    1. Visor del Último Mes  — tortas, KPIs de variación mensual e interanual
    2. Análisis Histórico    — series de tiempo con medias móviles, tabla, variaciones
    3. Proyecciones          — placeholder para auto-ARIMA (próxima versión)
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# CONSTANTES — mapeo de impuestos a filas del Excel (índice 0-based de fila)
# La fila 9 del Excel (índice 8 en 0-based) es la cabecera de fechas.
# Las filas de datos comienzan en la fila 11 (índice 10).
# ---------------------------------------------------------------------------

# Índice de fila en el DataFrame crudo (0-based, contando desde la fila 1 del Excel)
TAX_ROW_INDEX = {
    "Ganancias":                        12,   # Excel row 13
    "IVA":                              13,   # Excel row 14
    "Internos coparticipados":          15,   # Excel row 16
    "Bienes personales":                21,   # Excel row 22
    "Créditos y Débitos en cta. cte.":  22,   # Excel row 23
    "Combustibles Total":               25,   # Excel row 26
    "Monotributo impositivo":           29,   # Excel row 30
    "Derechos de importación":          36,   # Excel row 37
    "Derechos de exportación":          37,   # Excel row 38
    "Tasa de estadística":              38,   # Excel row 39
    "Seguridad Social":                 41,   # Excel row 42 (total AP. Y CONTRIB.)
    "Aportes personales":               43,   # Excel row 44
    "Contribuciones patronales":        44,   # Excel row 45
    "TOTAL REC. TRIBUTARIOS":           49,   # Excel row 50
}

# Lista de impuestos componentes del TOTAL (para torta, excluye el total mismo)
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

# Ruta al archivo Excel (relativa al directorio del script)
EXCEL_PATH = pathlib.Path(__file__).parent.parent.parent / (
    "attached_assets/Recaudación_2023_hasta_2026_Nominal_y_Real_1778714645756.xlsx"
)

# ---------------------------------------------------------------------------
# CARGA Y PARSEO DE DATOS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando datos desde Excel…")
def cargar_datos() -> dict:
    """
    Lee el archivo Excel y devuelve un diccionario con cuatro DataFrames,
    uno por cada pestaña: 'Nominal', 'Real', 'Var Nominal', 'Var Real'.

    Cada DataFrame tiene:
        - Índice: pd.DatetimeIndex con los meses
        - Columnas: nombre del impuesto (str)
    """
    if not EXCEL_PATH.exists():
        st.error(f"No se encontró el archivo Excel en: {EXCEL_PATH}")
        st.stop()

    raw_nom = pd.read_excel(EXCEL_PATH, sheet_name="Nominal", header=None)
    raw_real = pd.read_excel(EXCEL_PATH, sheet_name="Real",    header=None)

    def parsear_hoja(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Toma el DataFrame crudo (todas las celdas) y devuelve un DataFrame
        limpio con índice de fechas y columnas por impuesto.
        """
        # Fila 9 del Excel (índice 8) contiene las fechas en las columnas C en adelante (índice 2+)
        fechas = pd.to_datetime(raw.iloc[8, 2:], errors="coerce")
        mascara_fechas = fechas.notna()
        fechas = fechas[mascara_fechas]

        series_dict = {}
        for nombre, fila_idx in TAX_ROW_INDEX.items():
            fila = raw.iloc[fila_idx, 2:]          # columnas desde C en adelante
            fila = fila[mascara_fechas]             # alinear con fechas válidas
            valores = pd.to_numeric(fila, errors="coerce")
            serie = pd.Series(valores.values, index=fechas, name=nombre)
            series_dict[nombre] = serie

        return pd.DataFrame(series_dict)

    df_nom  = parsear_hoja(raw_nom)
    df_real = parsear_hoja(raw_real)

    # Variaciones mensuales: pct_change mes a mes
    df_var_nom  = df_nom.pct_change()
    df_var_real = df_real.pct_change()

    # Variaciones interanuales: pct_change de 12 períodos
    df_ia_nom  = df_nom.pct_change(12)
    df_ia_real = df_real.pct_change(12)

    return {
        "Nominal":       df_nom,
        "Real":          df_real,
        "Var Nominal":   df_var_nom,
        "Var Real":      df_var_real,
        "IA Nominal":    df_ia_nom,
        "IA Real":       df_ia_real,
    }


def filtrar_periodo(df: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """
    Filtra el DataFrame al período seleccionado.

    Períodos disponibles:
        '6 meses', '1 año', '2 años', 'Histórico'
    """
    if df.empty:
        return df

    ultimo_mes = df.index.max()

    if periodo == "6 meses":
        desde = ultimo_mes - pd.DateOffset(months=5)
    elif periodo == "1 año":
        desde = ultimo_mes - pd.DateOffset(months=11)
    elif periodo == "2 años":
        desde = ultimo_mes - pd.DateOffset(months=23)
    else:
        return df  # Histórico: sin filtro

    return df[df.index >= desde]


def formatear_millones(valor: float) -> str:
    """Formatea un número como millones de pesos con separadores de miles."""
    if pd.isna(valor):
        return "—"
    if abs(valor) >= 1_000_000:
        return f"${valor/1_000_000:,.1f}B"   # billones (millones de millones)
    return f"${valor:,.0f}M"


def formatear_porcentaje(valor: float) -> str:
    """Formatea un valor como porcentaje con signo."""
    if pd.isna(valor):
        return "—"
    signo = "+" if valor >= 0 else ""
    return f"{signo}{valor*100:.1f}%"


# ---------------------------------------------------------------------------
# COMPONENTES DE GRÁFICOS
# ---------------------------------------------------------------------------

def grafico_serie_temporal(
    serie: pd.Series,
    nombre_impuesto: str,
    tipo_recaudacion: str,
    periodo: str,
) -> go.Figure:
    """
    Grafico de línea con la serie histórica del impuesto seleccionado,
    con medias móviles de 3 y 6 meses superpuestas.
    """
    serie_filtrada = filtrar_periodo(serie.to_frame(), periodo)[nombre_impuesto]

    # Calcular medias móviles sobre la serie COMPLETA y luego filtrar
    # para que las primeras observaciones del período no sean NaN
    serie_completa = serie
    mm3  = serie_completa.rolling(window=3,  min_periods=1).mean()
    mm6  = serie_completa.rolling(window=6,  min_periods=1).mean()

    mm3_filtrada = filtrar_periodo(mm3.to_frame(name="mm3"),  periodo)["mm3"]
    mm6_filtrada = filtrar_periodo(mm6.to_frame(name="mm6"),  periodo)["mm6"]

    fig = go.Figure()

    # Línea principal
    fig.add_trace(go.Scatter(
        x=serie_filtrada.index,
        y=serie_filtrada.values,
        mode="lines+markers",
        name=nombre_impuesto,
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=5),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>",
    ))

    # Media móvil 3 meses
    fig.add_trace(go.Scatter(
        x=mm3_filtrada.index,
        y=mm3_filtrada.values,
        mode="lines",
        name="MM 3 meses",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        hovertemplate="<b>MM3 %{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>",
    ))

    # Media móvil 6 meses
    fig.add_trace(go.Scatter(
        x=mm6_filtrada.index,
        y=mm6_filtrada.values,
        mode="lines",
        name="MM 6 meses",
        line=dict(color="#2ca02c", width=2, dash="dash"),
        hovertemplate="<b>MM6 %{x|%b %Y}</b><br>%{y:,.0f} M$<extra></extra>",
    ))

    fig.update_layout(
        title=f"{nombre_impuesto} — {tipo_recaudacion} ({periodo})",
        xaxis_title="Mes",
        yaxis_title="Millones de pesos",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
        margin=dict(t=60, b=40, l=60, r=20),
    )

    return fig


def grafico_variacion_mensual(serie_var: pd.Series, nombre: str, periodo: str) -> go.Figure:
    """
    Gráfico de barras con la variación mensual (mes a mes) del impuesto seleccionado.
    Barras verdes para valores positivos, rojas para negativos.
    """
    data = filtrar_periodo(serie_var.to_frame(), periodo)[nombre].dropna()

    colores = ["#2ca02c" if v >= 0 else "#d62728" for v in data.values]

    fig = go.Figure(go.Bar(
        x=data.index,
        y=data.values * 100,
        marker_color=colores,
        hovertemplate="<b>%{x|%b %Y}</b><br>Var mensual: %{y:.1f}%<extra></extra>",
        name="Variación mensual",
    ))

    fig.add_hline(y=0, line_width=1, line_color="black")

    fig.update_layout(
        title=f"Variación mensual — {nombre}",
        xaxis_title="Mes",
        yaxis_title="Variación (%)",
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )

    return fig


def grafico_variacion_interanual(serie_ia: pd.Series, nombre: str, periodo: str) -> go.Figure:
    """
    Gráfico de barras con la variación interanual (año a año, 12 meses) del impuesto.
    Barras verdes para valores positivos, rojas para negativos.
    """
    data = filtrar_periodo(serie_ia.to_frame(), periodo)[nombre].dropna()

    colores = ["#2ca02c" if v >= 0 else "#d62728" for v in data.values]

    fig = go.Figure(go.Bar(
        x=data.index,
        y=data.values * 100,
        marker_color=colores,
        hovertemplate="<b>%{x|%b %Y}</b><br>Var interanual: %{y:.1f}%<extra></extra>",
        name="Variación interanual",
    ))

    fig.add_hline(y=0, line_width=1, line_color="black")

    fig.update_layout(
        title=f"Variación interanual — {nombre}",
        xaxis_title="Mes",
        yaxis_title="Variación i.a. (%)",
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )

    return fig


def grafico_torta(df_mes: pd.Series, tipo_recaudacion: str, fecha: pd.Timestamp) -> go.Figure:
    """
    Gráfico de torta con la composición de la recaudación total para un mes dado.
    Usa los componentes definidos en COMPONENTES_TORTA (excluye el TOTAL).
    """
    # Tomar solo los impuestos componentes y descartar nulos o negativos
    valores = []
    etiquetas = []
    for comp in COMPONENTES_TORTA:
        v = df_mes.get(comp, np.nan)
        if pd.notna(v) and v > 0:
            valores.append(v)
            etiquetas.append(comp)

    fig = go.Figure(go.Pie(
        labels=etiquetas,
        values=valores,
        hole=0.35,
        marker=dict(colors=PALETA_TORTA),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} M$<br>%{percent}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Composición de la Recaudación — {tipo_recaudacion} — {fecha.strftime('%B %Y')}",
        height=500,
        margin=dict(t=60, b=20, l=20, r=20),
        legend=dict(orientation="v", x=1.02, y=0.5),
    )

    return fig


def tabla_ultimos_6_meses(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """
    Devuelve un DataFrame con los últimos 6 meses del impuesto seleccionado,
    formateado para mostrar en pantalla.
    """
    serie = df[nombre].dropna().tail(6)

    tabla = pd.DataFrame({
        "Mes": [f.strftime("%B %Y") for f in serie.index],
        "Recaudación (M$)": [f"{v:,.0f}" for v in serie.values],
    })

    return tabla


# ---------------------------------------------------------------------------
# SIDEBAR — controles globales
# ---------------------------------------------------------------------------

def render_sidebar(datos: dict) -> tuple:
    """
    Renderiza el panel lateral con los controles del dashboard.

    Retorna:
        tipo_recaudacion (str): 'Nominal' o 'Real'
        impuesto (str): nombre del impuesto seleccionado
        periodo (str): período de análisis
    """
    st.sidebar.title("⚙️ Controles")
    st.sidebar.markdown("---")

    # Selector de tipo de recaudación (Nominal / Real)
    tipo_recaudacion = st.sidebar.radio(
        "Tipo de recaudación",
        options=["Nominal", "Real"],
        help="Nominal: valores corrientes. Real: ajustados por inflación (base 2023).",
    )

    # Selector de impuesto
    opciones_impuestos = list(TAX_ROW_INDEX.keys())
    impuesto = st.sidebar.selectbox(
        "Impuesto",
        options=opciones_impuestos,
        index=opciones_impuestos.index("TOTAL REC. TRIBUTARIOS"),
        help="Seleccioná el impuesto o agregado a visualizar.",
    )

    st.sidebar.markdown("---")

    # Selector de período
    periodo = st.sidebar.radio(
        "Período a visualizar",
        options=["6 meses", "1 año", "2 años", "Histórico"],
        index=1,
        help="Filtra el rango de fechas en los gráficos de serie temporal.",
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
    """
    Renderiza la sección 'Visor del Último Mes':
      - KPIs: recaudación total, variación mensual, variación interanual
      - Gráfico de torta con la composición por impuesto
      - Tabla resumen con todos los impuestos para el último mes
    """
    df      = datos[tipo_recaudacion]
    df_vm   = datos[f"Var {tipo_recaudacion}"]    # variación mensual
    df_ia   = datos[f"IA {tipo_recaudacion}"]      # variación interanual

    # Último mes con datos para el TOTAL
    ultimo_mes = df["TOTAL REC. TRIBUTARIOS"].dropna().index.max()
    fila_mes   = df.loc[ultimo_mes]

    total      = fila_mes["TOTAL REC. TRIBUTARIOS"]
    var_mens   = df_vm.loc[ultimo_mes, "TOTAL REC. TRIBUTARIOS"]
    var_ia     = df_ia.loc[ultimo_mes,  "TOTAL REC. TRIBUTARIOS"]

    st.subheader(f"📅 Datos del mes: **{ultimo_mes.strftime('%B %Y')}**")

    # --- KPIs principales ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Recaudación Total",
            value=formatear_millones(total),
            help="Recaudación del último mes disponible.",
        )

    with col2:
        delta_mens = formatear_porcentaje(var_mens)
        st.metric(
            label="Variación mensual",
            value=delta_mens,
            delta=delta_mens,
            delta_color="normal",
            help="Variación respecto al mes anterior.",
        )

    with col3:
        delta_ia = formatear_porcentaje(var_ia)
        st.metric(
            label="Variación interanual",
            value=delta_ia,
            delta=delta_ia,
            delta_color="normal",
            help="Variación respecto al mismo mes del año anterior.",
        )

    st.markdown("---")

    # --- Gráfico de torta ---
    fig_torta = grafico_torta(fila_mes, tipo_recaudacion, ultimo_mes)
    st.plotly_chart(fig_torta, width="stretch")

    # --- Tabla resumen de todos los impuestos ---
    st.subheader("Resumen por impuesto")

    rows = []
    for nombre in TAX_ROW_INDEX.keys():
        v     = fila_mes.get(nombre, np.nan)
        vm    = df_vm.loc[ultimo_mes, nombre] if nombre in df_vm.columns else np.nan
        via   = df_ia.loc[ultimo_mes,  nombre] if nombre in df_ia.columns else np.nan

        rows.append({
            "Impuesto":              nombre,
            "Recaudación (M$)":      f"{v:,.0f}" if pd.notna(v) else "—",
            "Var. mensual":          formatear_porcentaje(vm),
            "Var. interanual (i.a.)": formatear_porcentaje(via),
        })

    tabla_resumen = pd.DataFrame(rows)
    st.dataframe(tabla_resumen, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# SECCIÓN 2: Análisis Histórico
# ---------------------------------------------------------------------------

def render_historico(datos: dict, tipo_recaudacion: str, impuesto: str, periodo: str) -> None:
    """
    Renderiza la sección 'Análisis Histórico':
      - Gráfico de serie temporal con medias móviles 3 y 6 meses
      - Tabla de los últimos 6 meses
      - Gráfico de variación mensual
      - Gráfico de variación interanual
    """
    df    = datos[tipo_recaudacion]
    df_vm = datos[f"Var {tipo_recaudacion}"]
    df_ia = datos[f"IA {tipo_recaudacion}"]

    st.subheader(f"📈 {impuesto} — {tipo_recaudacion}")

    # --- Gráfico de serie temporal con medias móviles ---
    serie = df[impuesto]
    fig_ts = grafico_serie_temporal(serie, impuesto, tipo_recaudacion, periodo)
    st.plotly_chart(fig_ts, width="stretch")

    # --- Tabla últimos 6 meses ---
    col_tabla, col_info = st.columns([2, 3])

    with col_tabla:
        st.markdown("**Últimos 6 meses**")
        tabla = tabla_ultimos_6_meses(df, impuesto)
        st.dataframe(tabla, width="stretch", hide_index=True)

    with col_info:
        # Mini-KPIs de contexto
        serie_data = df[impuesto].dropna()
        if len(serie_data) >= 2:
            ult  = serie_data.iloc[-1]
            ant  = serie_data.iloc[-2]
            vm   = (ult / ant - 1) if ant != 0 else np.nan
            ia   = df_ia[impuesto].dropna()
            ia_v = ia.iloc[-1] if not ia.empty else np.nan

            st.markdown("**Indicadores del último mes**")
            k1, k2 = st.columns(2)
            k1.metric("Último valor", formatear_millones(ult))
            k2.metric("Var. mensual", formatear_porcentaje(vm),
                      delta=formatear_porcentaje(vm))

            k3, k4 = st.columns(2)
            k3.metric("Var. interanual", formatear_porcentaje(ia_v),
                      delta=formatear_porcentaje(ia_v))
            # Máximo histórico
            k4.metric("Máximo histórico", formatear_millones(serie_data.max()))

    st.markdown("---")

    # --- Gráficos de variación ---
    col_var1, col_var2 = st.columns(2)

    with col_var1:
        if impuesto in df_vm.columns:
            fig_vm = grafico_variacion_mensual(df_vm[impuesto], impuesto, periodo)
            st.plotly_chart(fig_vm, width="stretch")

    with col_var2:
        if impuesto in df_ia.columns:
            fig_ia = grafico_variacion_interanual(df_ia[impuesto], impuesto, periodo)
            st.plotly_chart(fig_ia, width="stretch")


# ---------------------------------------------------------------------------
# SECCIÓN 3: Proyecciones (placeholder)
# ---------------------------------------------------------------------------

def render_proyecciones() -> None:
    """
    Placeholder para la futura funcionalidad de proyección con Auto-ARIMA.
    """
    st.subheader("🔮 Proyecciones — Auto-ARIMA")

    st.info(
        "**Próximamente:** Esta sección implementará proyecciones automáticas "
        "usando el modelo Auto-ARIMA (pmdarima) para estimar la recaudación "
        "de los próximos 3, 6 y 12 meses por impuesto.\n\n"
        "**Funcionalidades planificadas:**\n"
        "- Selección del horizonte de proyección (1–12 meses)\n"
        "- Intervalo de confianza al 80% y 95%\n"
        "- Diagnósticos del modelo (residuos, ACF, PACF)\n"
        "- Exportación de proyecciones a Excel/CSV",
        icon="🚧",
    )

    st.markdown("---")
    st.markdown(
        "Para implementar esta funcionalidad instalá el paquete `pmdarima`:\n"
        "```bash\npip install pmdarima\n```"
    )


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

def main() -> None:
    """Función principal que orquesta el dashboard."""

    # Título y descripción general
    st.title("📊 Dashboard de Recaudación Tributaria")
    st.caption(
        "Análisis de la recaudación tributaria argentina. "
        "Fuente: datos internos de la oficina. Valores en millones de pesos."
    )

    # Cargar datos desde el Excel
    datos = cargar_datos()

    # Renderizar sidebar y obtener controles
    tipo_recaudacion, impuesto, periodo = render_sidebar(datos)

    # Tabs principales
    tab_mes, tab_hist, tab_proy = st.tabs([
        "📅 Visor del Último Mes",
        "📈 Análisis Histórico",
        "🔮 Proyecciones (próximamente)",
    ])

    with tab_mes:
        render_ultimo_mes(datos, tipo_recaudacion)

    with tab_hist:
        render_historico(datos, tipo_recaudacion, impuesto, periodo)

    with tab_proy:
        render_proyecciones()


if __name__ == "__main__":
    main()
