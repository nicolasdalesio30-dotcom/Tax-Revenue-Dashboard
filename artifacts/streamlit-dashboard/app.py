"""
Dashboard de Recaudación Tributaria
====================================
Dashboard interactivo para análisis y proyección de la recaudación tributaria argentina.

Módulos auxiliares:
    loader.py   — carga de datos (recaudación, macro, REM BCRA)
    forecast.py — modelos ARIMA, métricas y scoring
    ai_tools.py — resumen IA del Boletín Oficial + sentimiento macro
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from loader   import (cargar_datos, cargar_datos_macro, cargar_rem,
                      proyectar_real_con_rem, COMPONENTES_TORTA, TAX_ROW_INDEX)
from forecast import (pronosticar_serie_cache, pronosticar_macro_cache,
                      pronosticar_con_exogenas_cache, generar_fechas_futuras,
                      render_metricas_completas, calcular_mape, calcular_rmse)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recaudación Tributaria",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# HELPERS — MESES EN ESPAÑOL Y FORMATEO
# ---------------------------------------------------------------------------

MESES_CORTOS = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
MESES_LARGOS = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}


def fmt_mes(dt) -> str:
    """'May 2026' → 'May. 2026' en español."""
    return f"{MESES_CORTOS[dt.month]} {dt.year}"


def fmt_mes_largo(dt) -> str:
    return f"{MESES_LARGOS[dt.month]} {dt.year}"


def eje_x_espanol(fig: go.Figure, fechas, axis: str = "xaxis") -> go.Figure:
    """Aplica etiquetas de meses en español al eje X de un gráfico Plotly."""
    fechas = list(fechas)
    fig.update_layout(**{axis: dict(
        tickvals=fechas,
        ticktext=[fmt_mes(f) for f in fechas],
        tickangle=-40,
    )})
    return fig


def formatear_millones(v: float) -> str:
    if pd.isna(v): return "—"
    if abs(v) >= 1_000_000: return f"${v/1_000_000:,.1f}B"
    return f"${v:,.0f}M"


def formatear_pct(v: float, decimales: int = 1) -> str:
    if pd.isna(v): return "—"
    signo = "+" if v >= 0 else ""
    return f"{signo}{v*100:.{decimales}f}%"


# ---------------------------------------------------------------------------
# GRÁFICOS — con meses en español
# ---------------------------------------------------------------------------

PALETA_TORTA = px.colors.qualitative.Set3
COLOR_HIST   = "#1f77b4"
COLOR_FC_S   = "#ff7f0e"
COLOR_FC_M   = "#2ca02c"


def grafico_serie_temporal(serie, nombre, tipo, periodo):
    """Línea histórica + medias móviles 3M y 6M con meses en español."""
    def _filt(s):
        df = s.to_frame()
        ult = df.index.max()
        offsets = {"6 meses": 5, "1 año": 11, "2 años": 23}
        if periodo in offsets:
            df = df[df.index >= ult - pd.DateOffset(months=offsets[periodo])]
        return df.iloc[:, 0]

    sf  = _filt(serie)
    mm3 = _filt(serie.rolling(3, min_periods=1).mean())
    mm6 = _filt(serie.rolling(6, min_periods=1).mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sf.index, y=sf.values, mode="lines+markers",
        name=nombre, line=dict(color=COLOR_HIST, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|}</b><br>%{y:,.0f} M$<extra></extra>"))
    fig.add_trace(go.Scatter(x=mm3.index, y=mm3.values, mode="lines",
        name="MM 3m", line=dict(color="#ff7f0e", width=2, dash="dot"),
        hovertemplate="<b>%{x|}</b><br>MM3: %{y:,.0f} M$<extra></extra>"))
    fig.add_trace(go.Scatter(x=mm6.index, y=mm6.values, mode="lines",
        name="MM 6m", line=dict(color="#2ca02c", width=2, dash="dash"),
        hovertemplate="<b>%{x|}</b><br>MM6: %{y:,.0f} M$<extra></extra>"))
    fig.update_layout(
        title=f"{nombre} — {tipo} ({periodo})",
        xaxis_title="Mes", yaxis_title="Millones de $",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=430, margin=dict(t=60, b=50, l=60, r=20))
    eje_x_espanol(fig, sf.index)
    return fig


def grafico_barras_variacion(data: pd.Series, titulo: str, ylabel: str) -> go.Figure:
    colores = ["#2ca02c" if v >= 0 else "#d62728" for v in data.values]
    fig = go.Figure(go.Bar(x=data.index, y=data.values * 100, marker_color=colores,
        hovertemplate="<b>%{x|}</b><br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(title=titulo, xaxis_title="Mes", yaxis_title=ylabel,
        height=340, margin=dict(t=50, b=50, l=60, r=20))
    eje_x_espanol(fig, data.index)
    return fig


def grafico_torta(fila_mes, tipo, fecha):
    """
    Torta de composición. Solo muestra % dentro del slice;
    etiquetas completas van a la leyenda lateral.
    """
    vals, labs = [], []
    for comp in COMPONENTES_TORTA:
        v = fila_mes.get(comp, np.nan)
        if pd.notna(v) and v > 0:
            vals.append(v)
            labs.append(comp)

    fig = go.Figure(go.Pie(
        labels=labs, values=vals, hole=0.38,
        marker=dict(colors=PALETA_TORTA),
        textinfo="percent", texttemplate="%{percent:.1%}",
        insidetextorientation="radial", automargin=True,
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} M$<br>%{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Composición — {tipo} — {fmt_mes_largo(fecha)}",
        height=520, margin=dict(t=60, b=20, l=20, r=200),
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=12), title="Impuesto"),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return fig


def grafico_pronostico(serie_hist, fechas_fut, forecast, ci_lower, ci_upper,
                        titulo, nombre_serie="Histórico", nombre_fc="Pronóstico",
                        color_hist=COLOR_HIST, color_fc=COLOR_FC_S, n_hist=24):
    """Histórico + banda IC 80% + pronóstico en un único gráfico."""
    hist = serie_hist.dropna().tail(n_hist)
    fc   = np.array(forecast)
    lo   = np.array(ci_lower)
    hi   = np.array(ci_upper)

    # Color hex → RGB para la banda semitransparente
    def _hex_rgb(h): return int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    r, g, b = _hex_rgb(color_fc)

    fig = go.Figure()

    # Histórico
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers",
        name=nombre_serie, line=dict(color=color_hist, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|}</b><br>%{y:,.1f}<extra></extra>"))

    # Banda IC 80%
    fig.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 80%", hoverinfo="skip"))

    # Pronóstico
    fig.add_trace(go.Scatter(x=fechas_fut, y=fc, mode="lines+markers",
        name=nombre_fc, line=dict(color=color_fc, width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|}</b><br>%{y:,.1f}<extra></extra>"))

    # Línea vertical fin del histórico
    fig.add_vline(x=str(serie_hist.dropna().index.max()), line_dash="dot",
                  line_color="gray", line_width=1)

    fig.update_layout(
        title=titulo, xaxis_title="Mes", yaxis_title="Valor",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=430, margin=dict(t=70, b=50, l=60, r=20))

    todas_fechas = list(hist.index) + list(fechas_fut)
    eje_x_espanol(fig, todas_fechas)
    return fig


def grafico_heatmap_correlacion(df_macro: dict, serie_tax: pd.Series,
                                 nombre_tax: str, max_lag: int = 4) -> go.Figure:
    """
    Heatmap de correlación de Pearson entre cada variable macro y el impuesto,
    a diferentes rezagos (lag 0 … lag_max meses).
    """
    nombres_col = [f"Lag {i}m" for i in range(max_lag + 1)]
    filas = {}

    for nombre_macro, serie_macro in df_macro.items():
        if serie_macro is None or serie_macro.empty:
            continue
        corrs = []
        for lag in range(max_lag + 1):
            x = serie_macro.shift(lag)
            alineado = pd.concat([serie_tax, x], axis=1).dropna()
            if len(alineado) >= 8:
                c = float(alineado.iloc[:, 0].corr(alineado.iloc[:, 1]))
            else:
                c = np.nan
            corrs.append(round(c, 3) if not np.isnan(c) else np.nan)
        filas[nombre_macro] = corrs

    if not filas:
        return go.Figure()

    df_corr = pd.DataFrame(filas, index=nombres_col).T

    fig = go.Figure(go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns.tolist(),
        y=df_corr.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row]
              for row in df_corr.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> × %{x}<br>Correlación: %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", tickvals=[-1, -0.5, 0, 0.5, 1]),
    ))
    fig.update_layout(
        title=f"Correlación de Pearson: Variables Macro × {nombre_tax}",
        xaxis_title="Rezago (meses)", yaxis_title="Variable Macro",
        height=320, margin=dict(t=60, b=40, l=140, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar(datos: dict) -> tuple:
    st.sidebar.title("⚙️ Controles")
    st.sidebar.markdown("---")

    tipo = st.sidebar.radio(
        "Tipo de recaudación",
        options=["Nominal", "Real"],
        help="Nominal: valores corrientes. Real: deflactados por IPC (base 2023). "
             "Los meses sin IPC disponible se proyectan usando el REM del BCRA.",
    )

    opciones = list(TAX_ROW_INDEX.keys())
    impuesto = st.sidebar.selectbox(
        "Impuesto",
        options=opciones,
        index=opciones.index("TOTAL REC. TRIBUTARIOS"),
        help="Se aplica a todas las pestañas de análisis y pronóstico.",
    )

    st.sidebar.markdown("---")
    periodo = st.sidebar.radio(
        "Período histórico",
        options=["6 meses", "1 año", "2 años", "Histórico"],
        index=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Fuente: datos de oficina. Valores en millones de pesos corrientes o constantes."
    )
    return tipo, impuesto, periodo


# ---------------------------------------------------------------------------
# TAB 1 — Visor del Último Mes
# ---------------------------------------------------------------------------

def render_ultimo_mes(datos: dict, tipo: str) -> None:
    df    = datos[tipo]
    df_vm = datos[f"Var {tipo}"]
    df_ia = datos[f"IA {tipo}"]

    ult_mes  = df["TOTAL REC. TRIBUTARIOS"].dropna().index.max()
    fila_mes = df.loc[ult_mes]
    total    = fila_mes["TOTAL REC. TRIBUTARIOS"]
    vm_tot   = df_vm.loc[ult_mes, "TOTAL REC. TRIBUTARIOS"]
    ia_tot   = df_ia.loc[ult_mes,  "TOTAL REC. TRIBUTARIOS"]

    st.subheader(f"📅 Último mes disponible: **{fmt_mes_largo(ult_mes)}**")

    # --- IPC / inflación ---
    macro = cargar_datos_macro()
    ipc   = macro.get("IPC", pd.Series(dtype=float)).dropna()
    ipc_vm = ipc.pct_change()
    ult_ipc_mes = ipc.index.max()
    inflacion_mensual = ipc_vm.iloc[-1] if not ipc_vm.empty else np.nan
    inflacion_ia      = ipc.pct_change(12).iloc[-1] if len(ipc) >= 12 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recaudación Total",    formatear_millones(total))
    c2.metric("Var. mensual",         formatear_pct(vm_tot),
              delta=formatear_pct(vm_tot), delta_color="normal")
    c3.metric("Var. interanual",      formatear_pct(ia_tot),
              delta=formatear_pct(ia_tot), delta_color="normal")

    # "¿Cuánto dio la inflación?"
    c4.metric(
        f"Inflación mensual ({fmt_mes(ult_ipc_mes)})",
        formatear_pct(inflacion_mensual),
        delta=formatear_pct(inflacion_mensual),
        delta_color="inverse",
        help="Variación mensual del IPC INDEC. Dato más reciente disponible.",
    )
    c5.metric(
        "Inflación interanual",
        formatear_pct(inflacion_ia),
        delta=formatear_pct(inflacion_ia),
        delta_color="inverse",
        help="Variación del IPC respecto al mismo mes del año anterior.",
    )

    st.markdown("---")

    # --- Torta ---
    st.plotly_chart(grafico_torta(fila_mes, tipo, ult_mes), width="stretch")

    # --- Tabla resumen ---
    st.subheader("Resumen por impuesto")
    rows = []
    for nombre in TAX_ROW_INDEX:
        v   = fila_mes.get(nombre, np.nan)
        vm  = df_vm.loc[ult_mes, nombre] if nombre in df_vm.columns else np.nan
        via = df_ia.loc[ult_mes, nombre] if nombre in df_ia.columns else np.nan
        rows.append({
            "Impuesto":               nombre,
            "Recaudación (M$)":       f"{v:,.0f}" if pd.notna(v) else "—",
            "Var. mensual":           formatear_pct(vm),
            "Var. interanual (i.a.)": formatear_pct(via),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2 — Análisis Histórico
# ---------------------------------------------------------------------------

def render_historico(datos: dict, tipo: str, impuesto: str, periodo: str,
                     rem_serie: pd.Series, ipc_serie: pd.Series,
                     info_proyec: dict) -> None:
    df    = datos[tipo]
    df_vm = datos[f"Var {tipo}"]
    df_ia = datos[f"IA {tipo}"]

    st.subheader(f"📈 {impuesto} — {tipo}")

    # Si hay meses proyectados y el tipo es Real, mostrar aviso
    if tipo == "Real" and info_proyec:
        meses_proy = [fmt_mes_largo(m) for m in sorted(info_proyec.keys())]
        tasas_str  = ", ".join(f"{t:.1f}%" for t in
                               [info_proyec[m] for m in sorted(info_proyec.keys())])
        st.info(
            f"📌 Los datos reales de **{', '.join(meses_proy)}** se estimaron con inflación "
            f"proyectada del REM (BCRA): **{tasas_str} mensual** respectivamente.",
            icon="📌",
        )

    # Gráfico serie temporal + medias móviles
    st.plotly_chart(
        grafico_serie_temporal(df[impuesto], impuesto, tipo, periodo),
        width="stretch",
    )

    # Tabla últimos 6 meses + mini KPIs
    col_t, col_k = st.columns([2, 3])
    with col_t:
        st.markdown("**Últimos 6 meses**")
        sd = df[impuesto].dropna().tail(6)
        tabla = pd.DataFrame({
            "Mes": [fmt_mes_largo(f) for f in sd.index],
            "Recaudación (M$)": [f"{v:,.0f}" for v in sd.values],
        })
        st.dataframe(tabla, width="stretch", hide_index=True)

    with col_k:
        sd_full = df[impuesto].dropna()
        if len(sd_full) >= 2:
            ult, ant = sd_full.iloc[-1], sd_full.iloc[-2]
            vm_k     = (ult / ant - 1) if ant else np.nan
            ia_k     = df_ia[impuesto].dropna()
            ia_v     = ia_k.iloc[-1] if not ia_k.empty else np.nan
            st.markdown("**Indicadores del último mes**")
            k1, k2 = st.columns(2)
            k1.metric("Último valor",     formatear_millones(ult))
            k2.metric("Var. mensual",     formatear_pct(vm_k),
                      delta=formatear_pct(vm_k))
            k3, k4 = st.columns(2)
            k3.metric("Var. interanual",  formatear_pct(ia_v),
                      delta=formatear_pct(ia_v))
            k4.metric("Máximo histórico", formatear_millones(sd_full.max()))

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        def _filt(s):
            df2 = s.dropna().to_frame()
            ult = df2.index.max()
            offs = {"6 meses": 5, "1 año": 11, "2 años": 23}
            if periodo in offs:
                df2 = df2[df2.index >= ult - pd.DateOffset(months=offs[periodo])]
            return df2.iloc[:, 0]

        data_vm = _filt(df_vm[impuesto])
        st.plotly_chart(
            grafico_barras_variacion(data_vm, f"Variación mensual — {impuesto}", "Var. (%)"),
            width="stretch")
    with c2:
        data_ia = _filt(df_ia[impuesto])
        st.plotly_chart(
            grafico_barras_variacion(data_ia, f"Variación interanual — {impuesto}", "Var. i.a. (%)"),
            width="stretch")


# ---------------------------------------------------------------------------
# TAB 3 — Pronóstico Simple
# ---------------------------------------------------------------------------

def render_pronostico_simple(datos: dict, tipo: str, impuesto: str,
                              rem_serie: pd.Series, fecha_rem) -> None:
    st.subheader(f"🔮 Pronóstico Simple — {impuesto} ({tipo})")
    st.caption(
        "Modelo **Auto-ARIMA estacional (m=12)** entrenado solo con la serie histórica. "
        "Intervalo de confianza al 80%. Validación: últimos 3 meses como test."
    )

    df     = datos[tipo]
    serie  = df[impuesto].dropna()
    horiz  = 6

    # Mostrar si se usa REM para el IPC
    if not rem_serie.empty and fecha_rem is not None:
        st.info(
            f"📊 Las expectativas de inflación del REM (BCRA) al "
            f"**{fmt_mes_largo(fecha_rem)}** se usan como referencia adicional "
            f"para contextualizar el pronóstico de la serie real.",
            icon="📊",
        )

    with st.spinner("⏳ Entrenando Auto-ARIMA…"):
        res = pronosticar_serie_cache(
            vals=tuple(serie.values.tolist()),
            fechas_iso=tuple(serie.index.strftime("%Y-%m-%d").tolist()),
            horizonte=horiz, seasonal=True, m=12,
        )

    if "error" in res:
        st.error(f"Error al entrenar el modelo: {res['error']}")
        return

    fechas_fut = generar_fechas_futuras(serie.index.max(), horiz)

    st.plotly_chart(
        grafico_pronostico(
            serie_hist=serie, fechas_fut=fechas_fut,
            forecast=res["forecast"], ci_lower=res["ci_lower"], ci_upper=res["ci_upper"],
            titulo=f"{impuesto} — Pronóstico simple a {horiz} meses (IC 80%)",
        ),
        width="stretch",
    )

    # Tabla de valores proyectados
    df_fc = pd.DataFrame({
        "Mes":             [fmt_mes_largo(f) for f in fechas_fut],
        "Pronóstico (M$)": [f"{v:,.0f}" for v in res["forecast"]],
        "IC Inf. (80%)":   [f"{v:,.0f}" for v in res["ci_lower"]],
        "IC Sup. (80%)":   [f"{v:,.0f}" for v in res["ci_upper"]],
    })
    st.markdown("**Valores proyectados con intervalo de confianza al 80%**")
    st.dataframe(df_fc, hide_index=True)

    st.markdown("---")
    render_metricas_completas(res, float(serie.mean()),
                               titulo="📊 Métricas del modelo")


# ---------------------------------------------------------------------------
# TAB 4 — Pronóstico Macro
# ---------------------------------------------------------------------------

CONFIG_MACRO = {
    "IPC":   dict(label="IPC (Inflación mensual)", unidad="Índice", color="#e377c2"),
    "Dolar": dict(label="Tipo de Cambio — Dólar Oficial", unidad="$/USD", color="#bcbd22"),
    "Tasa":  dict(label="Tasa de Interés (depósitos 30d)", unidad="TNA (%)", color="#17becf"),
    "EMAE":  dict(label="EMAE (Actividad Económica)", unidad="Índice", color="#8c564b"),
}


def render_pronostico_macro() -> dict:
    """
    Muestra grilla 2×2 con histórico + pronóstico a 6 meses para cada macro.
    Incluye análisis de sentimiento de mercado vía IA.
    Retorna dict {nombre_macro: pd.Series de pronóstico futuro}.
    """
    st.subheader("📉 Proyección de Variables Macroeconómicas")
    st.caption(
        "Modelos **Auto-ARIMA** independientes para IPC, Dólar, Tasa y EMAE. "
        "Los pronósticos futuros se usan como exógenas en el Modelo con Macro."
    )

    macro = cargar_datos_macro()
    horiz = 6
    resultados = {}
    pron_futuros = {}

    with st.spinner("⏳ Entrenando modelos ARIMA para las 4 variables…"):
        for key in CONFIG_MACRO:
            serie = macro.get(key, pd.Series(dtype=float)).dropna()
            if serie.empty:
                resultados[key] = {"error": "Sin datos"}
                continue
            res = pronosticar_macro_cache(
                vals=tuple(serie.values.tolist()),
                fechas_iso=tuple(serie.index.strftime("%Y-%m-%d").tolist()),
                horizonte=horiz,
            )
            resultados[key] = res
            if "error" not in res:
                fechas_fut = generar_fechas_futuras(serie.index.max(), horiz)
                pron_futuros[key] = pd.Series(res["forecast"], index=fechas_fut)

    # --- Grilla 2×2 de gráficos ---
    keys = list(CONFIG_MACRO.keys())
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for j, key in enumerate(keys[i: i + 2]):
            with cols[j]:
                cfg   = CONFIG_MACRO[key]
                res   = resultados.get(key, {})
                serie = macro.get(key, pd.Series(dtype=float)).dropna()

                if "error" in res:
                    st.error(f"{cfg['label']}: {res['error']}")
                    continue

                fechas_fut = generar_fechas_futuras(serie.index.max(), horiz)
                fig = grafico_pronostico(
                    serie_hist=serie, fechas_fut=fechas_fut,
                    forecast=res["forecast"],
                    ci_lower=res["ci_lower"], ci_upper=res["ci_upper"],
                    titulo=cfg["label"], color_fc=cfg["color"], n_hist=30,
                )
                fig.update_layout(yaxis_title=cfg["unidad"], height=370)
                st.plotly_chart(fig, width="stretch")

                # Variación del último mes
                if len(serie) >= 2:
                    var_ult = serie.iloc[-1] / serie.iloc[-2] - 1
                else:
                    var_ult = np.nan

                m1, m2, m3 = st.columns(3)
                m1.metric("AIC",    f"{res['aic']:.1f}" if not np.isnan(res["aic"]) else "—")
                m2.metric("Orden",  res.get("orden", "—"))
                m3.metric("Var. últ. mes", formatear_pct(var_ult),
                          delta=formatear_pct(var_ult),
                          delta_color="inverse" if key in ("IPC","Dolar","Tasa") else "normal")

                # Sentimiento de IA (expandible)
                with st.expander(f"🧠 Análisis de sentimiento — {cfg['label']}"):
                    mes_ult  = serie.index.max()
                    with st.spinner("Consultando IA…"):
                        from ai_tools import analizar_sentimiento_macro
                        sent = analizar_sentimiento_macro(
                            variable=key,
                            mes=mes_ult.month, anio=mes_ult.year,
                            ultimo_valor=float(serie.iloc[-1]),
                            variacion_pct=float(var_ult) * 100 if not np.isnan(var_ult) else np.nan,
                        )
                    st.markdown(sent)

    st.markdown("---")
    st.success(
        "✅ Pronósticos macro listos. Pasá a **Modelo con Macro** para la proyección integrada."
    )
    return pron_futuros


# ---------------------------------------------------------------------------
# TAB 5 — Modelo con Macro (ARIMAX)
# ---------------------------------------------------------------------------

def render_modelo_con_macro(datos: dict, tipo: str, impuesto: str) -> None:
    st.subheader("🧠 Modelo con Variables Macroeconómicas")
    st.info(
        "Las proyecciones de las variables macro se obtienen de modelos ARIMA individuales. "
        "El modelo con exógenas incorpora estos pronósticos para mejorar la predicción "
        "de la recaudación.",
        icon="ℹ️",
    )

    df    = datos[tipo]
    y     = df[impuesto].dropna()
    macro = cargar_datos_macro()
    horiz = 6

    macro_disp = {k: v.dropna() for k, v in macro.items()
                  if v is not None and not v.empty}

    fecha_inicio = max(y.index.min(), *[s.index.min() for s in macro_disp.values()])
    fecha_fin    = min(y.index.max(), *[s.index.max() for s in macro_disp.values()])
    y_alin       = y[(y.index >= fecha_inicio) & (y.index <= fecha_fin)]

    X_hist_dict = {}
    for key, serie in macro_disp.items():
        s = serie.reindex(y_alin.index, method="ffill").ffill().bfill()
        if s.isna().sum() < len(s) * 0.5:
            X_hist_dict[key] = s.values.tolist()

    if not X_hist_dict:
        st.error("No hay macros disponibles para alinear.")
        return

    # Pronósticos futuros de las macros
    X_fut_dict = {}
    with st.spinner("⏳ Computando pronósticos macro…"):
        for key, serie in macro_disp.items():
            if key not in X_hist_dict:
                continue
            rm = pronosticar_macro_cache(
                vals=tuple(serie.values.tolist()),
                fechas_iso=tuple(serie.index.strftime("%Y-%m-%d").tolist()),
                horizonte=horiz,
            )
            if "error" not in rm:
                X_fut_dict[key] = rm["forecast"]

    macros_comunes = [k for k in X_hist_dict if k in X_fut_dict]
    X_h = {k: X_hist_dict[k] for k in macros_comunes}
    X_f = {k: X_fut_dict[k]  for k in macros_comunes}

    # Modelo simple (referencia)
    with st.spinner("⏳ Entrenando modelo simple (referencia)…"):
        res_s = pronosticar_serie_cache(
            vals=tuple(y_alin.values.tolist()),
            fechas_iso=tuple(y_alin.index.strftime("%Y-%m-%d").tolist()),
            horizonte=horiz, seasonal=True, m=12,
        )

    # Modelo con exógenas
    with st.spinner("⏳ Entrenando ARIMAX con macros…"):
        res_m = pronosticar_con_exogenas_cache(
            y_vals=tuple(y_alin.values.tolist()),
            y_fechas=tuple(y_alin.index.strftime("%Y-%m-%d").tolist()),
            X_hist_dict=X_h, X_fut_dict=X_f, horizonte=horiz,
        )

    if "error" in res_m or "error" in res_s:
        st.error(f"Error en el modelo: {res_m.get('error','')}{res_s.get('error','')}")
        return

    fechas_fut = generar_fechas_futuras(y_alin.index.max(), horiz)

    # --- Gráfico comparativo ---
    hist_plot = y_alin.tail(24)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot.values,
        mode="lines+markers", name="Histórico",
        line=dict(color=COLOR_HIST, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|}</b><br>%{y:,.0f} M$<extra></extra>"))

    # IC simple
    fig.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=res_s["ci_upper"] + res_s["ci_lower"][::-1],
        fill="toself", fillcolor="rgba(255,127,14,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="IC Simple (80%)", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=fechas_fut, y=res_s["forecast"],
        mode="lines+markers", name="Simple (ARIMA)",
        line=dict(color=COLOR_FC_S, width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|}</b><br>%{y:,.0f} M$<extra></extra>"))

    # IC macro
    fig.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=res_m["ci_upper"] + res_m["ci_lower"][::-1],
        fill="toself", fillcolor="rgba(44,160,44,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="IC Macro (80%)", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=fechas_fut, y=res_m["forecast"],
        mode="lines+markers", name=f"Con Macro ({', '.join(macros_comunes)})",
        line=dict(color=COLOR_FC_M, width=2.5, dash="dot"),
        marker=dict(size=7, symbol="circle"),
        hovertemplate="<b>%{x|}</b><br>%{y:,.0f} M$<extra></extra>"))

    fig.add_vline(x=str(y_alin.index.max()), line_dash="dot",
                  line_color="gray", line_width=1)
    fig.update_layout(
        title=f"{impuesto} — Simple vs. ARIMAX (IC 80%)",
        xaxis_title="Mes", yaxis_title="Millones de $",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=490, margin=dict(t=80, b=50, l=60, r=20))
    eje_x_espanol(fig, list(hist_plot.index) + list(fechas_fut))
    st.plotly_chart(fig, width="stretch")

    # --- Tabla comparativa ---
    st.markdown("**Valores proyectados mes a mes**")
    df_cmp = pd.DataFrame({
        "Mes":           [fmt_mes_largo(f) for f in fechas_fut],
        "Simple (M$)":   [f"{v:,.0f}" for v in res_s["forecast"]],
        "Con Macro (M$)":[f"{v:,.0f}" for v in res_m["forecast"]],
        "Diferencia":    [f"{(m-s):+,.0f}" for s, m in
                          zip(res_s["forecast"], res_m["forecast"])],
    })
    st.dataframe(df_cmp, hide_index=True)

    st.markdown("---")

    # --- Heatmap de correlación ---
    st.markdown("**📊 Correlación de Pearson entre variables macro y el impuesto seleccionado**")
    fig_hm = grafico_heatmap_correlacion(macro_disp, y_alin, impuesto, max_lag=4)
    st.plotly_chart(fig_hm, width="stretch")
    st.caption(
        "Lag 0m: correlación contemporánea. Lag Nm: la macro desplazada N meses predice "
        "la recaudación actual. Valores > 0.5 o < -0.5 indican relación significativa."
    )

    st.markdown("---")

    # --- Métricas comparativas ---
    st.markdown("**Comparación de métricas (validación en últimos 3 meses)**")
    media_y = float(y_alin.mean())

    col_s, col_m = st.columns(2)
    with col_s:
        render_metricas_completas(res_s, media_y, titulo="Modelo Simple")
    with col_m:
        render_metricas_completas(res_m, media_y, titulo="Modelo con Macro")

    # Mejora MAPE
    mape_s = res_s.get("mape", np.nan)
    mape_m = res_m.get("mape", np.nan)
    if not np.isnan(mape_s) and not np.isnan(mape_m) and mape_s != 0:
        mejora = (mape_s - mape_m) / mape_s * 100
        color  = "#1a7f37" if mejora > 0 else "#cf222e"
        st.markdown(
            f"**Mejora MAPE del modelo con macro vs. simple:** "
            f'<span style="color:{color};font-weight:700">{mejora:+.1f}%</span>',
            unsafe_allow_html=True,
        )

    with st.expander("ℹ️ Variables exógenas utilizadas"):
        st.markdown(
            "| Variable | Descripción |\n|---|---|\n"
            + "\n".join(
                f"| **{k}** | {CONFIG_MACRO[k]['label']} |"
                for k in macros_comunes
            )
        )
        st.markdown(
            f"**Período de entrenamiento:** "
            f"{fmt_mes_largo(y_alin.index.min())} → {fmt_mes_largo(y_alin.index.max())} "
            f"({len(y_alin)} obs.)"
        )


# ---------------------------------------------------------------------------
# TAB 6 — Boletín Oficial + IA
# ---------------------------------------------------------------------------

def render_boletin_oficial(impuesto: str) -> None:
    st.subheader("📰 Boletín Oficial y Novedades del Impuesto")
    st.caption(
        "Generá un resumen de las publicaciones del Boletín Oficial de la República Argentina "
        "y noticias financieras relevantes para el impuesto seleccionado en el período indicado."
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        mes_sel  = st.selectbox("Mes",
            options=list(range(1, 13)),
            format_func=lambda m: MESES_LARGOS[m],
            index=0,
        )
    with col_f2:
        anio_sel = st.selectbox("Año",
            options=list(range(2023, 2027)),
            index=2,
        )

    if st.button("🔍 Buscar y resumir", type="primary"):
        with st.spinner(
            f"Buscando publicaciones sobre **{impuesto}** en "
            f"**{MESES_LARGOS[mes_sel]} {anio_sel}**…"
        ):
            from ai_tools import resumir_boletin_oficial
            resumen = resumir_boletin_oficial(
                impuesto=impuesto, mes=mes_sel, anio=anio_sel,
            )

        st.markdown("---")
        st.markdown(
            f"### Resumen — {impuesto} | {MESES_LARGOS[mes_sel]} {anio_sel}"
        )
        st.markdown(resumen)
        st.markdown("---")
        st.markdown(
            "🔗 **Verificar publicaciones directamente:** "
            "[boletinoficial.gob.ar](https://www.boletinoficial.gob.ar/busquedaAvanzada)"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📊 Dashboard de Recaudación Tributaria")
    st.caption(
        "Análisis y proyección de la recaudación tributaria argentina. "
        "Fuente: datos internos de la oficina. Valores en millones de pesos."
    )

    # --- Carga de datos ---
    datos     = cargar_datos()
    rem_serie, fecha_rem = cargar_rem()
    macro     = cargar_datos_macro()
    ipc_serie = macro.get("IPC", pd.Series(dtype=float)).dropna()

    # --- Extensión de la serie Real con REM ---
    df_real_ext, info_proyec = proyectar_real_con_rem(
        datos["Nominal"], datos["Real"], ipc_serie, rem_serie,
    )
    datos["Real"] = df_real_ext
    # Recalcular variaciones con la serie extendida
    datos["Var Real"] = df_real_ext.pct_change()
    datos["IA Real"]  = df_real_ext.pct_change(12)

    # --- Sidebar ---
    tipo, impuesto, periodo = render_sidebar(datos)

    # --- Tabs ---
    (tab_mes, tab_hist, tab_simple,
     tab_macro, tab_exog, tab_boletin) = st.tabs([
        "📅 Último Mes",
        "📈 Análisis Histórico",
        "🔮 Pronóstico Simple",
        "📉 Pronóstico Macro",
        "🧠 Modelo con Macro",
        "📰 Boletín Oficial",
    ])

    with tab_mes:
        render_ultimo_mes(datos, tipo)

    with tab_hist:
        render_historico(datos, tipo, impuesto, periodo,
                         rem_serie, ipc_serie, info_proyec)

    with tab_simple:
        render_pronostico_simple(datos, tipo, impuesto, rem_serie, fecha_rem)

    with tab_macro:
        render_pronostico_macro()

    with tab_exog:
        render_modelo_con_macro(datos, tipo, impuesto)

    with tab_boletin:
        render_boletin_oficial(impuesto)


if __name__ == "__main__":
    main()
