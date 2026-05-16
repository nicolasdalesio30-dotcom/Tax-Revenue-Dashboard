"""
Dashboard de Recaudación Tributaria
=====================================
Herramienta permanente de análisis y proyección para la oficina.

Módulos:
    loader.py             — carga de datos (recaudación, macro, REM BCRA)
    forecast.py           — modelos ARIMA, métricas, scoring, transparencia
    tests_estadisticos.py — Kruskal-Wallis + Chow, pre-tests para ARIMA
    ai_tools.py           — scraping web real + Claude (BOA, sentimiento)
"""

import hashlib
import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from loader import (
    cargar_datos, cargar_datos_macro, cargar_rem,
    proyectar_real_con_rem, COMPONENTES_TORTA, TAX_ROW_INDEX,
)
from forecast import (
    pronosticar_serie_cache, pronosticar_macro_cache,
    pronosticar_con_exogenas_cache, generar_fechas_futuras,
    render_metricas_completas, render_backend_modelo,
    calcular_mape, calcular_rmse,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recaudación Tributaria",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# HELPERS — MESES EN ESPAÑOL
# ---------------------------------------------------------------------------

MESES_CORTOS = {
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
    7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic",
}
MESES_LARGOS = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
    7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre",
}


def fmt_mes(dt) -> str:
    return f"{MESES_CORTOS[dt.month]} {dt.year}"


def fmt_mes_largo(dt) -> str:
    return f"{MESES_LARGOS[dt.month]} {dt.year}"


def eje_x_espanol(fig, fechas, axis="xaxis"):
    fechas = list(fechas)
    fig.update_layout(**{axis: dict(
        tickvals=fechas,
        ticktext=[fmt_mes(f) for f in fechas],
        tickangle=-40,
    )})
    return fig


def formatear_millones(v):
    if pd.isna(v): return "—"
    if abs(v) >= 1_000_000: return f"${v/1_000_000:,.1f}B"
    return f"${v:,.0f}M"


def formatear_pct(v, decimales=1):
    if pd.isna(v): return "—"
    signo = "+" if v >= 0 else ""
    return f"{signo}{v*100:.{decimales}f}%"


# ---------------------------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------------------------

COLOR_HIST  = "#1f77b4"
COLOR_FC_S  = "#ff7f0e"
COLOR_FC_M  = "#2ca02c"
PALETA_TORTA = px.colors.qualitative.Set3

CONFIG_MACRO = {
    "IPC":   dict(label="IPC (Inflación mensual)", unidad="Índice", color="#e377c2"),
    "Dolar": dict(label="Tipo de Cambio — Dólar Oficial", unidad="$/USD", color="#bcbd22"),
    "Tasa":  dict(label="Tasa de Interés (depósitos 30d)", unidad="TNA (%)", color="#17becf"),
    "EMAE":  dict(label="EMAE (Actividad Económica)", unidad="Índice", color="#8c564b"),
}


def grafico_serie_temporal(serie, nombre, tipo, periodo):
    def _filt(s):
        df = s.to_frame()
        ult = df.index.max()
        offs = {"6 meses": 5, "1 año": 11, "2 años": 23}
        if periodo in offs:
            df = df[df.index >= ult - pd.DateOffset(months=offs[periodo])]
        return df.iloc[:, 0]

    sf  = _filt(serie)
    mm3 = _filt(serie.rolling(3, min_periods=1).mean())
    mm6 = _filt(serie.rolling(6, min_periods=1).mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sf.index, y=sf.values, mode="lines+markers",
        name=nombre, line=dict(color=COLOR_HIST, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|}</b><br>%{y:,.0f} M$<extra></extra>"))
    fig.add_trace(go.Scatter(x=mm3.index, y=mm3.values, mode="lines",
        name="MM 3m", line=dict(color="#ff7f0e", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=mm6.index, y=mm6.values, mode="lines",
        name="MM 6m", line=dict(color="#2ca02c", width=2, dash="dash")))
    fig.update_layout(
        title=f"{nombre} — {tipo} ({periodo})", xaxis_title="Mes",
        yaxis_title="Millones de $", hovermode="x unified", height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=50, l=60, r=20))
    eje_x_espanol(fig, sf.index)
    return fig


def grafico_barras_variacion(data, titulo, ylabel):
    colores = ["#2ca02c" if v >= 0 else "#d62728" for v in data.values]
    fig = go.Figure(go.Bar(
        x=data.index, y=data.values * 100, marker_color=colores,
        hovertemplate="<b>%{x|}</b><br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(title=titulo, xaxis_title="Mes", yaxis_title=ylabel,
        height=340, margin=dict(t=50, b=50, l=60, r=20))
    eje_x_espanol(fig, data.index)
    return fig


def grafico_torta(fila_mes, tipo, fecha):
    vals, labs = [], []
    for comp in COMPONENTES_TORTA:
        v = fila_mes.get(comp, np.nan)
        if pd.notna(v) and v > 0:
            vals.append(v)
            labs.append(comp)
    fig = go.Figure(go.Pie(
        labels=labs, values=vals, hole=0.38,
        marker=dict(colors=PALETA_TORTA), textinfo="percent",
        texttemplate="%{percent:.1%}", insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} M$<br>%{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Composición — {tipo} — {fmt_mes_largo(fecha)}",
        height=520, margin=dict(t=60, b=20, l=20, r=200),
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=12)),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return fig


def grafico_pronostico(serie_hist, fechas_fut, forecast, ci_lower, ci_upper,
                        titulo, nombre_serie="Histórico", nombre_fc="Pronóstico",
                        color_hist=COLOR_HIST, color_fc=COLOR_FC_S, n_hist=24):
    hist = serie_hist.dropna().tail(n_hist)
    fc   = np.array(forecast)
    lo   = np.array(ci_lower)
    hi   = np.array(ci_upper)

    def _hex_rgb(h): return int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    r, g, b = _hex_rgb(color_fc)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers",
        name=nombre_serie, line=dict(color=color_hist, width=2.5), marker=dict(size=5),
        hovertemplate="<b>%{x|}</b><br>%{y:,.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=list(fechas_fut) + list(fechas_fut[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 80%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fechas_fut, y=fc, mode="lines+markers",
        name=nombre_fc, line=dict(color=color_fc, width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="<b>%{x|}</b><br>%{y:,.1f}<extra></extra>"))
    fig.add_vline(x=str(serie_hist.dropna().index.max()), line_dash="dot",
                  line_color="gray", line_width=1)
    fig.update_layout(
        title=titulo, xaxis_title="Mes", yaxis_title="Valor",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=430, margin=dict(t=70, b=50, l=60, r=20))
    eje_x_espanol(fig, list(hist.index) + list(fechas_fut))
    return fig


def grafico_heatmap_correlacion(macro_dict, serie_tax, nombre_tax, max_lag=4):
    nombres_col = [f"Lag {i}m" for i in range(max_lag + 1)]
    filas = {}
    for nombre_macro, serie_macro in macro_dict.items():
        if serie_macro is None or serie_macro.empty:
            continue
        # Alinear índices con la serie del impuesto
        common_idx = serie_tax.index.intersection(serie_macro.index)
        if len(common_idx) < 8:
            continue  # pocos datos, no se puede calcular
        serie_macro_alin = serie_macro.reindex(common_idx)
        serie_tax_alin = serie_tax.reindex(common_idx)
        corrs = []
        for lag in range(max_lag + 1):
            x = serie_macro_alin.shift(lag)
            alin = pd.concat([serie_tax_alin, x], axis=1).dropna()
            if len(alin) >= 8:
                corrs.append(round(float(alin.iloc[:,0].corr(alin.iloc[:,1])), 3))
            else:
                corrs.append(np.nan)
        filas[nombre_macro] = corrs

    if not filas:
        fig = go.Figure()
        fig.add_annotation(text="Datos insuficientes para calcular correlaciones", x=0.5, y=0.5, showarrow=False)
        return fig

    df_corr = pd.DataFrame(filas, index=nombres_col).T
    fig = go.Figure(go.Heatmap(
        z=df_corr.values, x=df_corr.columns.tolist(), y=df_corr.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in df_corr.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> × %{x}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ", tickvals=[-1, -0.5, 0, 0.5, 1]),
    ))
    fig.update_layout(
        title=f"Correlación de Pearson: Variables Macro × {nombre_tax}",
        xaxis_title="Rezago (meses)", yaxis_title="Variable Macro",
        height=330, margin=dict(t=60, b=40, l=150, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR — Controles + Carga de archivos
# ---------------------------------------------------------------------------

def render_sidebar(datos: dict) -> tuple:
    st.sidebar.title("⚙️ Controles")
    st.sidebar.markdown("---")

    tipo = st.sidebar.radio("Tipo de recaudación", ["Nominal", "Real"],
        help="Real: deflactado por IPC base 2023. Meses sin IPC proyectados con REM.")
    opciones = list(TAX_ROW_INDEX.keys())
    impuesto = st.sidebar.selectbox(
        "Impuesto", options=opciones,
        index=opciones.index("TOTAL REC. TRIBUTARIOS"))
    periodo = st.sidebar.radio(
        "Período histórico", ["6 meses", "1 año", "2 años", "Histórico"], index=1)

    # --- Carga de archivos ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Actualizar datos")
    st.sidebar.caption(
        "Subí nuevos Excels con el mismo formato para actualizar la base. "
        "El sistema detecta automáticamente el cambio y limpia el caché."
    )

    with st.sidebar.expander("📤 Subir nuevos archivos"):
        up_reca  = st.file_uploader(
            "Excel recaudación (.xlsx)", type=["xlsx"], key="up_reca",
            help="Mismo formato que el archivo original: sheets Nominal y Real, "
                 "fila 9 con fechas, columna B con nombres de impuestos."
        )
        up_macro = st.file_uploader(
            "Excel macro (.xlsx)", type=["xlsx"], key="up_macro",
            help="Sheets: IPC, Dolar oficial, Tasa de interes, EMAE."
        )
        up_rem   = st.file_uploader(
            "REM BCRA (.xlsx)", type=["xlsx"], key="up_rem",
            help="Excel histórico del REM del BCRA. "
                 "Si no subís nada, se descarga automáticamente del BCRA."
        )

        if st.button("✅ Aplicar archivos nuevos", type="primary"):
            actualizado = False
            if up_reca is not None:
                b = up_reca.read()
                h = hashlib.md5(b).hexdigest()
                if st.session_state.get("reca_hash") != h:
                    st.session_state["reca_bytes"] = b
                    st.session_state["reca_hash"]  = h
                    actualizado = True

            if up_macro is not None:
                b = up_macro.read()
                h = hashlib.md5(b).hexdigest()
                if st.session_state.get("macro_hash") != h:
                    st.session_state["macro_bytes"] = b
                    st.session_state["macro_hash"]  = h
                    actualizado = True

            if up_rem is not None:
                b = up_rem.read()
                h = hashlib.md5(b).hexdigest()
                if st.session_state.get("rem_hash") != h:
                    st.session_state["rem_bytes"] = b
                    st.session_state["rem_hash"]  = h
                    actualizado = True

            if actualizado:
                st.cache_data.clear()
                st.success("✅ Datos actualizados. Recargando…")
                st.rerun()
            else:
                st.info("No se detectaron cambios en los archivos.")

        # Estado actual de los datos cargados
        st.markdown("---")
        fuente_reca  = "📤 Subido" if "reca_bytes"  in st.session_state else "💾 Disco"
        fuente_macro = "📤 Subido" if "macro_bytes" in st.session_state else "💾 Disco"
        fuente_rem   = "📤 Subido" if "rem_bytes"   in st.session_state else "🌐 BCRA"
        st.caption(
            f"**Recaudación:** {fuente_reca} | "
            f"**Macro:** {fuente_macro} | "
            f"**REM:** {fuente_rem}"
        )

    return tipo, impuesto, periodo


# ---------------------------------------------------------------------------
# TAB 1 — Visor del Último Mes
# ---------------------------------------------------------------------------

def render_ultimo_mes(datos, tipo):
    df    = datos[tipo]
    df_vm = datos[f"Var {tipo}"]
    df_ia = datos[f"IA {tipo}"]

    ult_mes  = df["TOTAL REC. TRIBUTARIOS"].dropna().index.max()
    fila_mes = df.loc[ult_mes]
    total    = fila_mes["TOTAL REC. TRIBUTARIOS"]
    vm_tot   = df_vm.loc[ult_mes, "TOTAL REC. TRIBUTARIOS"]
    ia_tot   = df_ia.loc[ult_mes,  "TOTAL REC. TRIBUTARIOS"]

    st.subheader(f"📅 Último mes disponible: **{fmt_mes_largo(ult_mes)}**")

    macro = cargar_datos_macro(st.session_state.get("macro_bytes"))
    ipc   = macro.get("IPC", pd.Series(dtype=float)).dropna()
    ipc_vm = ipc.pct_change()
    ult_ipc_mes = ipc.index.max()
    inflacion_mensual = float(ipc_vm.iloc[-1]) if not ipc_vm.empty else np.nan
    inflacion_ia      = float(ipc.pct_change(12).iloc[-1]) if len(ipc) >= 12 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recaudación Total",    formatear_millones(total))
    c2.metric("Var. mensual",         formatear_pct(vm_tot),
              delta=formatear_pct(vm_tot), delta_color="normal")
    c3.metric("Var. interanual",      formatear_pct(ia_tot),
              delta=formatear_pct(ia_tot), delta_color="normal")
    c4.metric(
        f"🔥 Inflación mensual ({fmt_mes(ult_ipc_mes)})",
        formatear_pct(inflacion_mensual),
        delta=formatear_pct(inflacion_mensual), delta_color="inverse",
        help="Variación mensual del IPC INDEC. 'Cuánto dio la inflación' el último mes.",
    )
    c5.metric(
        "Inflación interanual",
        formatear_pct(inflacion_ia),
        delta=formatear_pct(inflacion_ia), delta_color="inverse",
    )

    st.markdown("---")
    st.plotly_chart(grafico_torta(fila_mes, tipo, ult_mes), width="stretch")

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

def render_historico(datos, tipo, impuesto, periodo, rem_serie, ipc_serie, info_proyec):
    df    = datos[tipo]
    df_vm = datos[f"Var {tipo}"]
    df_ia = datos[f"IA {tipo}"]

    st.subheader(f"📈 {impuesto} — {tipo}")

    if tipo == "Real" and info_proyec:
        meses_proy = [fmt_mes_largo(m) for m in sorted(info_proyec.keys())]
        tasas_str  = ", ".join(
            f"{info_proyec[m]:.1f}%" for m in sorted(info_proyec.keys())
        )
        st.info(
            f"📌 Los datos reales de **{', '.join(meses_proy)}** se estimaron "
            f"con inflación proyectada del REM (BCRA): **{tasas_str} mensual** "
            f"respectivamente.",
            icon="📌",
        )

    st.plotly_chart(
        grafico_serie_temporal(df[impuesto], impuesto, tipo, periodo),
        width="stretch",
    )

    col_t, col_k = st.columns([2, 3])
    with col_t:
        st.markdown("**Últimos 6 meses**")
        sd = df[impuesto].dropna().tail(6)
        st.dataframe(pd.DataFrame({
            "Mes": [fmt_mes_largo(f) for f in sd.index],
            "Recaudación (M$)": [f"{v:,.0f}" for v in sd.values],
        }), width="stretch", hide_index=True)

    with col_k:
        sd_full = df[impuesto].dropna()
        if len(sd_full) >= 2:
            ult, ant = sd_full.iloc[-1], sd_full.iloc[-2]
            vm_k = (ult / ant - 1) if ant else np.nan
            ia_k = df_ia[impuesto].dropna()
            ia_v = ia_k.iloc[-1] if not ia_k.empty else np.nan
            st.markdown("**Indicadores del último mes**")
            k1, k2 = st.columns(2)
            k1.metric("Último valor",    formatear_millones(ult))
            k2.metric("Var. mensual",    formatear_pct(vm_k), delta=formatear_pct(vm_k))
            k3, k4 = st.columns(2)
            k3.metric("Var. interanual", formatear_pct(ia_v), delta=formatear_pct(ia_v))
            k4.metric("Máximo histórico", formatear_millones(sd_full.max()))

    st.markdown("---")
    c1, c2 = st.columns(2)

    def _filt(s):
        df2 = s.dropna().to_frame()
        ult = df2.index.max()
        offs = {"6 meses": 5, "1 año": 11, "2 años": 23}
        if periodo in offs:
            df2 = df2[df2.index >= ult - pd.DateOffset(months=offs[periodo])]
        return df2.iloc[:, 0]

    with c1:
        st.plotly_chart(
            grafico_barras_variacion(_filt(df_vm[impuesto]),
                                     f"Variación mensual — {impuesto}", "Var. (%)"),
            width="stretch")
    with c2:
        st.plotly_chart(
            grafico_barras_variacion(_filt(df_ia[impuesto]),
                                     f"Variación interanual — {impuesto}", "Var. i.a. (%)"),
            width="stretch")


# ---------------------------------------------------------------------------
# TAB 3 — Pronóstico Simple (con pre-tests)
# ---------------------------------------------------------------------------

def render_pronostico_simple(datos, tipo, impuesto, rem_serie, fecha_rem, df_rem_ui):
    st.subheader(f"🔮 Pronóstico Simple — {impuesto} ({tipo})")
    st.caption(
        "**Auto-ARIMA estacional (m=12)** entrenado solo con la serie histórica. "
        "IC al 80%. Validación en los últimos 3 meses."
    )

    df    = datos[tipo]
    serie = df[impuesto].dropna()
    horiz = 6

    # --- Pre-tests ---
    from tests_estadisticos import render_pretests
    config_tests = render_pretests(serie, impuesto)
    seasonal     = config_tests["seasonal"]
    m_arima      = config_tests["m"]
    fecha_inicio = config_tests["fecha_inicio"]
    fi_iso       = fecha_inicio.strftime("%Y-%m-%d") if fecha_inicio else None

    st.markdown("---")

    # --- REM info ---
    if not rem_serie.empty and fecha_rem is not None:
        with st.expander(
            f"📊 REM BCRA — Inflación esperada (corte: {fecha_rem.strftime('%b %Y')})"
        ):
            st.caption(
                "Medianas del Relevamiento de Expectativas de Mercado del BCRA. "
                "Estos valores se usan para proyectar la serie Real en meses sin IPC."
            )
            if not df_rem_ui.empty:
                st.dataframe(df_rem_ui.tail(12), hide_index=True, width="stretch")

    # --- Entrenar modelo ---
    with st.spinner("⏳ Entrenando Auto-ARIMA…"):
        res = pronosticar_serie_cache(
            vals=tuple(serie.values.tolist()),
            fechas_iso=tuple(serie.index.strftime("%Y-%m-%d").tolist()),
            horizonte=horiz, seasonal=seasonal, m=m_arima,
            fecha_inicio_iso=fi_iso,
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

    # Tabla valores proyectados
    df_fc = pd.DataFrame({
        "Mes":             [fmt_mes_largo(f) for f in fechas_fut],
        "Pronóstico (M$)": [f"{v:,.0f}" for v in res["forecast"]],
        "IC Inf. (80%)":   [f"{v:,.0f}" for v in res["ci_lower"]],
        "IC Sup. (80%)":   [f"{v:,.0f}" for v in res["ci_upper"]],
    })
    st.markdown("**Valores proyectados con intervalo de confianza al 80%**")
    st.dataframe(df_fc, hide_index=True)

    st.markdown("---")
    render_metricas_completas(res, float(serie.mean()), titulo="📊 Métricas del modelo")

    # Backend del modelo
    render_backend_modelo(
        serie=serie, res=res, fechas_fut=fechas_fut,
        config_pretests=config_tests, nombre=impuesto,
    )


# ---------------------------------------------------------------------------
# TAB 4 — Pronóstico Macro
# ---------------------------------------------------------------------------

def render_pronostico_macro():
    st.subheader("📉 Proyección de Variables Macroeconómicas")
    st.caption(
        "Modelos **Auto-ARIMA** independientes para IPC, Dólar, Tasa y EMAE. "
        "Los pronósticos futuros se usan como exógenas en el Modelo con Macro."
    )

    macro  = cargar_datos_macro(st.session_state.get("macro_bytes"))
    horiz  = 6
    pron_f = {}

    with st.spinner("⏳ Entrenando modelos ARIMA para las 4 variables…"):
        resultados = {}
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
                ff = generar_fechas_futuras(serie.index.max(), horiz)
                pron_f[key] = pd.Series(res["forecast"], index=ff)

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

                var_ult = float(serie.iloc[-1]/serie.iloc[-2] - 1) if len(serie) >= 2 else np.nan
                m1, m2, m3 = st.columns(3)
                m1.metric("AIC",    f"{res['aic']:.1f}" if not np.isnan(res["aic"]) else "—")
                m2.metric("Orden",  res.get("orden", "—"))
                m3.metric("Var. últ. mes", formatear_pct(var_ult),
                          delta=formatear_pct(var_ult),
                          delta_color="inverse" if key in ("IPC","Dolar","Tasa") else "normal")

                with st.expander(f"🧠 Sentimiento de mercado — {cfg['label']}"):
                    mes_ult = serie.index.max()
                    with st.spinner("Buscando noticias y analizando…"):
                        from ai_tools import analizar_sentimiento_macro
                        sent_data = analizar_sentimiento_macro(
                            variable=key,
                            mes=mes_ult.month, anio=mes_ult.year,
                            ultimo_valor=float(serie.iloc[-1]),
                            variacion_pct=var_ult * 100 if not np.isnan(var_ult) else np.nan,
                        )
                    st.markdown(sent_data["resumen"])
                    if sent_data.get("fuentes"):
                        st.markdown("**Fuentes consultadas:**")
                        for f in sent_data["fuentes"][:4]:
                            url  = f.get("url","")
                            title = f.get("title","Sin título")
                            date  = f.get("date","")
                            st.markdown(
                                f"- [{title}]({url})"
                                + (f" — {date}" if date else ""),
                                unsafe_allow_html=False,
                            )

    st.markdown("---")
    st.success("✅ Pronósticos macro listos. Pasá a **Modelo con Macro** para la proyección integrada.")
    return pron_f


# ---------------------------------------------------------------------------
# TAB 5 — Modelo con Macro (ARIMAX)
# ---------------------------------------------------------------------------

def render_modelo_con_macro(datos, tipo, impuesto):
    st.subheader("🧠 Modelo con Variables Macroeconómicas")
    st.info(
        "ARIMAX: el modelo incorpora las proyecciones de macro como variables explicativas. "
        "Los pre-tests determinan si se usa estacionalidad y qué rango temporal se entrena.",
        icon="ℹ️",
    )

    df    = datos[tipo]
    y     = df[impuesto].dropna()
    macro = cargar_datos_macro(st.session_state.get("macro_bytes"))
    horiz = 6

    macro_disp = {k: v.dropna() for k, v in macro.items() if v is not None and not v.empty}

    fecha_inicio_all = max(y.index.min(), *[s.index.min() for s in macro_disp.values()])
    fecha_fin_all    = min(y.index.max(), *[s.index.max() for s in macro_disp.values()])
    y_alin = y[(y.index >= fecha_inicio_all) & (y.index <= fecha_fin_all)]

    # --- Pre-tests en pestaña de macro ---
    from tests_estadisticos import render_pretests
    config_tests = render_pretests(y_alin, f"{impuesto}_macro")
    seasonal     = config_tests["seasonal"]
    fecha_inicio = config_tests["fecha_inicio"]
    fi_iso       = fecha_inicio.strftime("%Y-%m-%d") if fecha_inicio else None

    st.markdown("---")

    X_hist_dict = {}
    for key, serie in macro_disp.items():
        s = serie.reindex(y_alin.index, method="ffill").ffill().bfill()
        if s.isna().sum() < len(s) * 0.5:
            X_hist_dict[key] = s.values.tolist()

    # Pronósticos macro para el futuro
    X_fut_dict = {}
    with st.spinner("⏳ Computando pronósticos macro…"):
        for key, serie in macro_disp.items():
            if key not in X_hist_dict: continue
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

    # Modelos
    with st.spinner("⏳ Entrenando modelo simple (referencia)…"):
        res_s = pronosticar_serie_cache(
            vals=tuple(y_alin.values.tolist()),
            fechas_iso=tuple(y_alin.index.strftime("%Y-%m-%d").tolist()),
            horizonte=horiz, seasonal=seasonal, m=12 if seasonal else 1,
            fecha_inicio_iso=fi_iso,
        )

    with st.spinner("⏳ Entrenando ARIMAX con macros…"):
        res_m = pronosticar_con_exogenas_cache(
            y_vals=tuple(y_alin.values.tolist()),
            y_fechas=tuple(y_alin.index.strftime("%Y-%m-%d").tolist()),
            X_hist_dict=X_h, X_fut_dict=X_f,
            horizonte=horiz, fecha_inicio_iso=fi_iso,
        )

    if "error" in res_m or "error" in res_s:
        st.error(f"Error: {res_m.get('error','')} {res_s.get('error','')}")
        return

    fechas_fut  = generar_fechas_futuras(y_alin.index.max(), horiz)
    hist_plot   = y_alin.tail(24)

    # --- Gráfico comparativo ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot.values,
        mode="lines+markers", name="Histórico",
        line=dict(color=COLOR_HIST, width=2.5), marker=dict(size=5)))

    fig.add_trace(go.Scatter(
        x=list(fechas_fut)+list(fechas_fut[::-1]),
        y=res_s["ci_upper"]+res_s["ci_lower"][::-1],
        fill="toself", fillcolor="rgba(255,127,14,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="IC Simple 80%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fechas_fut, y=res_s["forecast"],
        mode="lines+markers", name="Simple (ARIMA)",
        line=dict(color=COLOR_FC_S, width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond")))

    fig.add_trace(go.Scatter(
        x=list(fechas_fut)+list(fechas_fut[::-1]),
        y=res_m["ci_upper"]+res_m["ci_lower"][::-1],
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="IC Macro 80%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fechas_fut, y=res_m["forecast"],
        mode="lines+markers",
        name=f"Con Macro ({', '.join(macros_comunes)})",
        line=dict(color=COLOR_FC_M, width=2.5, dash="dot"),
        marker=dict(size=7)))

    fig.add_vline(x=str(y_alin.index.max()), line_dash="dot",
                  line_color="gray", line_width=1)
    fig.update_layout(
        title=f"{impuesto} — Simple vs. ARIMAX (IC 80%)",
        xaxis_title="Mes", yaxis_title="Millones de $",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=500, margin=dict(t=80, b=50, l=60, r=20))
    eje_x_espanol(fig, list(hist_plot.index) + list(fechas_fut))
    st.plotly_chart(fig, width="stretch")

    # Tabla comparativa
    st.markdown("**Valores proyectados mes a mes**")
    df_cmp = pd.DataFrame({
        "Mes":            [fmt_mes_largo(f) for f in fechas_fut],
        "Simple (M$)":    [f"{v:,.0f}" for v in res_s["forecast"]],
        "Con Macro (M$)": [f"{v:,.0f}" for v in res_m["forecast"]],
        "Diferencia":     [f"{(m-s):+,.0f}" for s, m in
                           zip(res_s["forecast"], res_m["forecast"])],
    })
    st.dataframe(df_cmp, hide_index=True)

    st.markdown("---")

    # Heatmap correlaciones
    st.markdown("**📊 Correlación de Pearson: Variables Macro × Impuesto**")
    fig_hm = grafico_heatmap_correlacion(macro_disp, y_alin, impuesto, max_lag=4)
    st.plotly_chart(fig_hm, width="stretch")
    st.caption(
        "Lag 0m: contemporánea. Lag Nm: la macro desplazada N meses anticipa la recaudación. "
        "|ρ| > 0.5 indica relación relevante."
    )

    st.markdown("---")

    # Métricas comparativas
    media_y = float(y_alin.mean())
    col_s, col_m = st.columns(2)
    with col_s:
        render_metricas_completas(res_s, media_y, titulo="Modelo Simple")
    with col_m:
        render_metricas_completas(res_m, media_y, titulo="Modelo con Macro")

    mape_s = res_s.get("mape", np.nan)
    mape_m = res_m.get("mape", np.nan)
    if not np.isnan(mape_s) and not np.isnan(mape_m) and mape_s != 0:
        mejora = (mape_s - mape_m) / mape_s * 100
        color  = "#1a7f37" if mejora > 0 else "#cf222e"
        st.markdown(
            f"**Mejora MAPE (macro vs. simple):** "
            f'<span style="color:{color};font-weight:700">{mejora:+.1f}%</span>',
            unsafe_allow_html=True,
        )

    # Backend del modelo
    render_backend_modelo(
        serie=y_alin, res=res_m, fechas_fut=fechas_fut,
        config_pretests=config_tests, nombre=f"{impuesto}_macro",
    )


# ---------------------------------------------------------------------------
# TAB 6 — Boletín Oficial + IA (scraping real)
# ---------------------------------------------------------------------------

def render_boletin_oficial(impuesto):
    st.subheader("📰 Boletín Oficial y Novedades del Impuesto")
    st.caption(
        "Busca noticias reales en sitios financieros argentinos (Infobae, Ámbito, Cronista, "
        "AFIP, etc.) mediante scraping web en tiempo real. No usa el conocimiento del modelo — "
        "todo el contenido proviene de fuentes encontradas en el momento de la búsqueda."
    )

    st.info(
        "🌐 **Fuentes en tiempo real**: el sistema busca en la web y descarga el contenido "
        "de los artículos antes de resumirlos. Cuanto más reciente sea el período elegido, "
        "más probable es encontrar cobertura.",
        icon="🌐",
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        mes_sel = st.selectbox(
            "Mes", options=list(range(1, 13)),
            format_func=lambda m: MESES_LARGOS[m], index=0,
            key="boletin_mes",
        )
    with col_f2:
        anio_sel = st.selectbox(
            "Año", options=list(range(2023, 2027)), index=2,
            key="boletin_anio",
        )

    if st.button("🔍 Buscar y resumir en tiempo real", type="primary"):
        with st.spinner(
            f"Buscando publicaciones sobre **{impuesto}** en "
            f"**{MESES_LARGOS[mes_sel]} {anio_sel}**… "
            f"(puede tardar 20-40 seg.)"
        ):
            from ai_tools import resumir_boletin_con_ia
            resultado = resumir_boletin_con_ia(
                impuesto=impuesto, mes=mes_sel, anio=anio_sel,
            )

        st.markdown("---")
        n_f = resultado.get("n_fuentes", 0)
        if n_f == 0:
            st.warning(
                "⚠️ No se encontraron artículos específicos para este impuesto/período. "
                "El resumen puede ser más genérico."
            )
        else:
            st.success(
                f"✅ Se consultaron **{n_f} fuentes web** "
                f"(búsqueda realizada el {resultado.get('fecha_busqueda','')})"
            )

        st.markdown(
            f"### Resumen — {impuesto} | "
            f"{MESES_LARGOS[mes_sel]} {anio_sel}"
        )
        st.markdown(resultado["resumen"])

        # Fuentes
        fuentes = resultado.get("fuentes", [])
        if fuentes:
            st.markdown("---")
            st.markdown("**📎 Fuentes consultadas:**")
            for f in fuentes:
                url   = f.get("url", "")
                title = f.get("title", "Sin título")
                date  = f.get("date", "")
                if url:
                    st.markdown(
                        f"- [{title}]({url})"
                        + (f" — {date}" if date else "")
                    )
                else:
                    st.markdown(f"- {title}")

        st.markdown("---")
        st.markdown(
            "🔗 **Verificar publicaciones directamente en el BOA:** "
            "[boletinoficial.gob.ar](https://www.boletinoficial.gob.ar/busquedaAvanzada)"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📊 Dashboard de Recaudación Tributaria")
    st.caption(
        "Herramienta de análisis y proyección para la oficina. "
        "Subí nuevos Excels desde el panel lateral para actualizar todos los datos."
    )

    # Carga de datos (con soporte para archivos subidos)
    reca_bytes  = st.session_state.get("reca_bytes")
    macro_bytes = st.session_state.get("macro_bytes")
    rem_bytes   = st.session_state.get("rem_bytes")

    datos                = cargar_datos(reca_bytes)
    macro                = cargar_datos_macro(macro_bytes)
    rem_serie, fecha_rem, df_rem_ui = cargar_rem(rem_bytes)
    ipc_serie            = macro.get("IPC", pd.Series(dtype=float)).dropna()

    # Extensión serie Real con REM
    df_real_ext, info_proyec = proyectar_real_con_rem(
        datos["Nominal"], datos["Real"], ipc_serie, rem_serie,
    )
    datos["Real"]      = df_real_ext
    datos["Var Real"]  = df_real_ext.pct_change()
    datos["IA Real"]   = df_real_ext.pct_change(12)

    # Sidebar
    tipo, impuesto, periodo = render_sidebar(datos)

    # Tabs
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
        render_historico(datos, tipo, impuesto, periodo, rem_serie, ipc_serie, info_proyec)
    with tab_simple:
        render_pronostico_simple(datos, tipo, impuesto, rem_serie, fecha_rem, df_rem_ui)
    with tab_macro:
        render_pronostico_macro()
    with tab_exog:
        render_modelo_con_macro(datos, tipo, impuesto)
    with tab_boletin:
        render_boletin_oficial(impuesto)


if __name__ == "__main__":
    main()
