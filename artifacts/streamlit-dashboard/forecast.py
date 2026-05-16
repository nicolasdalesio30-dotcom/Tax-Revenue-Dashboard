"""
forecast.py — Modelos ARIMA, métricas, scoring y transparencia
===============================================================
Lógica econométrica completa:
  - ajuste de modelos Auto-ARIMA / ARIMAX
  - cálculo de errores (MAPE, RMSE, CV)
  - benchmarks visuales con colores
  - puntaje unificado del modelo (0–100)
  - sección de transparencia/backend (datos usados, validación, formula)
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

MESES_CORTOS = {
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
    7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic",
}


# ---------------------------------------------------------------------------
# HELPERS DE CÁLCULO
# ---------------------------------------------------------------------------

def calcular_mape(real: np.ndarray, pred: np.ndarray) -> float:
    mask = (real != 0) & np.isfinite(real) & np.isfinite(pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100)


def calcular_rmse(real: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(real) & np.isfinite(pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((real[mask] - pred[mask]) ** 2)))


def interpretar_arima(orden: tuple, seasonal_order: tuple = None) -> str:
    """
    Devuelve una interpretación en lenguaje llano del orden ARIMA.
    """
    p, d, q = orden
    partes = []
    if p > 0:
        partes.append(
            f"usa los **últimos {p} valor{'es' if p > 1 else ''}** "
            f"de la serie (componente AR={p})"
        )
    if d > 0:
        partes.append(
            f"aplica **{d} diferenciaci{'ones' if d > 1 else 'ón'}** "
            f"para hacer la serie estacionaria (I={d})"
        )
    if q > 0:
        partes.append(
            f"incorpora los **últimos {q} error{'es' if q > 1 else ''}** "
            f"de pronóstico (componente MA={q})"
        )
    if not partes:
        partes = ["modelo sin parámetros (random walk)"]

    texto = "El modelo " + ", ".join(partes) + "."

    if seasonal_order and any(s != 0 for s in seasonal_order[:3]):
        P, D, Q = seasonal_order[:3]
        texto += (
            f" Adicionalmente, incluye componente **estacional de 12 meses** "
            f"(SAR={P}, SD={D}, SMA={Q})."
        )
    return texto


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def calcular_score_modelo(mape: float, rmse: float, lb_pvalue: float,
                           media_y: float) -> int:
    def _p_mape(v):
        if np.isnan(v): return 50
        if v < 2:  return 95
        if v < 5:  return 82
        if v < 10: return 65
        if v < 20: return 45
        if v < 35: return 25
        return 8

    def _p_lb(p):
        if np.isnan(p): return 50
        if p > 0.20: return 95
        if p > 0.10: return 80
        if p > 0.05: return 65
        if p > 0.01: return 40
        return 12

    def _p_cv(r, m):
        if np.isnan(r) or np.isnan(m) or m == 0: return 50
        cv = r / abs(m) * 100
        if cv < 5:  return 95
        if cv < 10: return 82
        if cv < 20: return 65
        if cv < 35: return 45
        return 20

    return round(0.40 * _p_mape(mape) + 0.35 * _p_lb(lb_pvalue)
                 + 0.25 * _p_cv(rmse, media_y))


def etiqueta_score(score: int) -> tuple:
    if score >= 80: return "Excelente", "#1a7f37"
    if score >= 65: return "Bueno",     "#0969da"
    if score >= 50: return "Aceptable", "#bf8700"
    if score >= 35: return "Débil",     "#d14900"
    return "Revisar", "#cf222e"


# ---------------------------------------------------------------------------
# BENCHMARKS VISUALES
# ---------------------------------------------------------------------------

def _badge(texto: str, color: str) -> str:
    return (f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:0.85em;font-weight:600">{texto}</span>')


def _calidad_mape(v):
    if np.isnan(v): return "Sin datos", "#8b949e"
    if v < 2:  return "Excelente (<2%)",  "#1a7f37"
    if v < 5:  return "Bueno (<5%)",      "#0969da"
    if v < 10: return "Aceptable (<10%)", "#bf8700"
    if v < 20: return "Débil (<20%)",     "#d14900"
    return "Bajo (>20%)", "#cf222e"


def _calidad_ljung(p):
    if np.isnan(p): return "Sin datos", "#8b949e"
    if p > 0.20: return "Excelente (p>0.20)", "#1a7f37"
    if p > 0.10: return "Bueno (p>0.10)",     "#0969da"
    if p > 0.05: return "Aceptable (p>0.05)", "#bf8700"
    if p > 0.01: return "Débil (p>0.01)",     "#d14900"
    return "Autocorrelación detectada", "#cf222e"


def _calidad_cv(rmse, media):
    if np.isnan(rmse) or media == 0: return "Sin datos", "#8b949e"
    cv = rmse / abs(media) * 100
    if cv < 5:  return f"Excelente ({cv:.1f}%)", "#1a7f37"
    if cv < 10: return f"Bueno ({cv:.1f}%)",     "#0969da"
    if cv < 20: return f"Aceptable ({cv:.1f}%)", "#bf8700"
    if cv < 35: return f"Débil ({cv:.1f}%)",     "#d14900"
    return f"Bajo ({cv:.1f}%)", "#cf222e"


def render_metricas_completas(res: dict, media_y: float, titulo: str = "") -> None:
    mape  = res.get("mape",      np.nan)
    rmse  = res.get("rmse",      np.nan)
    lb_p  = res.get("lb_pvalue", np.nan)
    aic   = res.get("aic",       np.nan)
    bic   = res.get("bic",       np.nan)
    orden = res.get("orden",     "—")

    score              = calcular_score_modelo(mape, rmse, lb_p, media_y)
    etiq_score, c_sc   = etiqueta_score(score)

    if titulo:
        st.markdown(f"**{titulo}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAPE",  f"{mape:.1f}%" if not np.isnan(mape) else "—",
              help="Error porcentual medio absoluto (últimos 3 meses como test)")
    c2.metric("RMSE",  f"{rmse:,.0f}" if not np.isnan(rmse) else "—",
              help="Raíz del error cuadrático medio (M$)")
    c3.metric("AIC",   f"{aic:.1f}"   if not np.isnan(aic)  else "—",
              help="Criterio de Akaike — menor es mejor entre modelos")
    c4.metric("BIC",   f"{bic:.1f}"   if not np.isnan(bic)  else "—",
              help="Criterio Bayesiano — penaliza complejidad")
    c5.metric("Ljung-Box p",
              f"{lb_p:.3f}" if not np.isnan(lb_p) else "—",
              help="p>0.05 → residuos sin autocorrelación")

    b1, b2, b3, b4 = st.columns(4)
    etq_m, c_m = _calidad_mape(mape)
    etq_l, c_l = _calidad_ljung(lb_p)
    etq_c, c_c = _calidad_cv(rmse, media_y)

    with b1:
        st.markdown("**MAPE**")
        st.markdown(_badge(etq_m, c_m), unsafe_allow_html=True)
    with b2:
        st.markdown("**Ljung-Box**")
        st.markdown(_badge(etq_l, c_l), unsafe_allow_html=True)
    with b3:
        st.markdown("**RMSE / Media**")
        st.markdown(_badge(etq_c, c_c), unsafe_allow_html=True)
    with b4:
        st.markdown("**Puntaje del modelo**")
        st.markdown(
            f'<div style="font-size:2em;font-weight:700;color:{c_sc}">{score}/100</div>'
            f'<div>{_badge(etiq_score, c_sc)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<small>Orden ARIMA: <code>{orden}</code></small>",
        unsafe_allow_html=True,
    )

    with st.expander("📖 ¿Cómo interpretar estas métricas?"):
        st.markdown("""
| Métrica | Descripción | Rango ideal |
|---|---|---|
| **MAPE** | Error porcentual medio. Qué tan lejos está el pronóstico del valor real en %. | <5% excelente, <10% aceptable |
| **RMSE** | Error en millones de pesos. Comparar con el valor típico de la variable. | Depende de la escala |
| **RMSE / Media** | El RMSE como % del valor promedio. Permite comparar entre impuestos. | <10% bueno |
| **AIC / BIC** | Penalizan modelos complejos. Útiles solo para comparar modelos entre sí. | Menor es mejor |
| **Ljung-Box** | Prueba si los residuos son "ruido blanco" (el modelo capturó toda la señal). | p > 0.05 |
| **Puntaje** | Índice 0–100: MAPE (40%) + Ljung-Box (35%) + RMSE/Media (25%). | ≥65 confiable |

**Rangos del puntaje:**
🟢 80–100: Excelente | 🔵 65–79: Bueno | 🟡 50–64: Aceptable | 🟠 35–49: Débil | 🔴 0–34: Revisar
        """)


# ---------------------------------------------------------------------------
# BACKEND / TRANSPARENCIA DEL MODELO
# ---------------------------------------------------------------------------

def render_backend_modelo(
    serie: pd.Series, res: dict, fechas_fut,
    config_pretests: dict = None, nombre: str = "",
    X_hist: pd.DataFrame = None,
) -> None:
    """
    Sección expandible '🔬 Backend del modelo' con:
      - Datos de entrenamiento (tabla)
      - Validación: real vs pronosticado (últimos 3 meses)
      - Especificación del modelo y fórmula interpretada
      - Resultados de los pre-tests (si están disponibles)
      - Variables exógenas usadas (si las hay)
    """
    with st.expander("🔬 Datos utilizados y metodología del modelo"):
        t1, t2, t3 = st.tabs(["📋 Datos de entrenamiento", "🎯 Validación", "🧮 Metodología"])

        # --- TAB 1: Datos de entrenamiento ---
        with t1:
            fecha_inicio_str = ""
            if config_pretests and "fecha_inicio" in config_pretests:
                fi = config_pretests["fecha_inicio"]
                serie_train = serie[serie.index >= fi].dropna()
                fecha_inicio_str = fi.strftime("%b %Y")
            else:
                serie_train = serie.dropna()

            st.markdown(
                f"**Serie utilizada para entrenar el modelo**"
                + (f" (desde {fecha_inicio_str})" if fecha_inicio_str else "")
            )
            tabla_train = pd.DataFrame({
                "Mes": [f"{MESES_CORTOS[d.month]} {d.year}" for d in serie_train.index],
                "Valor (M$)": [f"{v:,.0f}" for v in serie_train.values],
                "Var. mensual": [
                    f"{((a/b)-1)*100:+.1f}%" if i > 0 and not np.isnan(b) and b != 0
                    else "—"
                    for i, (a, b) in enumerate(zip(
                        serie_train.values,
                        [np.nan] + serie_train.values[:-1].tolist()
                    ))
                ],
            })
            st.dataframe(tabla_train, width="stretch", hide_index=True)
            st.caption(
                f"**{len(serie_train)} observaciones** usadas para el entrenamiento. "
                f"Período: {tabla_train['Mes'].iloc[0]} → {tabla_train['Mes'].iloc[-1]}"
            )

            if X_hist is not None and not X_hist.empty:
                st.markdown("**Variables exógenas utilizadas**")
                st.dataframe(X_hist, width="stretch")

        # --- TAB 2: Validación ---
        with t2:
            y_test  = res.get("y_test",  [])
            y_pred  = res.get("y_pred_test", [])
            f_test  = res.get("fechas_test", [])

            if y_test and y_pred and len(y_test) == len(y_pred):
                tabla_val = pd.DataFrame({
                    "Mes": [
                        f"{MESES_CORTOS[pd.Timestamp(f).month]} {pd.Timestamp(f).year}"
                        for f in f_test
                    ] if f_test else [f"Mes -{i}" for i in range(len(y_test), 0, -1)],
                    "Real (M$)":        [f"{v:,.0f}" for v in y_test],
                    "Pronosticado (M$)":[f"{v:,.0f}" for v in y_pred],
                    "Error absoluto (M$)": [f"{abs(r-p):,.0f}" for r, p in zip(y_test, y_pred)],
                    "Error (%)": [
                        f"{abs(r-p)/abs(r)*100:.1f}%" if r != 0 else "—"
                        for r, p in zip(y_test, y_pred)
                    ],
                })
                st.markdown("**Validación en los últimos 3 meses (test fuera de muestra)**")
                st.dataframe(tabla_val, hide_index=True)

                mape_v = res.get("mape", np.nan)
                rmse_v = res.get("rmse", np.nan)
                cv, cr = st.columns(2)
                cv.metric("MAPE (validación)", f"{mape_v:.1f}%" if not np.isnan(mape_v) else "—")
                cr.metric("RMSE (validación)", f"{rmse_v:,.0f}" if not np.isnan(rmse_v) else "—")

                # Mini gráfico real vs pronosticado
                if f_test:
                    fig_val = go.Figure()
                    fechas_ts = [pd.Timestamp(f) for f in f_test]
                    fig_val.add_trace(go.Bar(
                        x=[f"{MESES_CORTOS[f.month]} {f.year}" for f in fechas_ts],
                        y=[abs(r - p) for r, p in zip(y_test, y_pred)],
                        name="Error absoluto (M$)", marker_color="#e377c2",
                    ))
                    fig_val.update_layout(
                        title="Error de validación por mes", height=250,
                        margin=dict(t=40, b=30, l=40, r=10),
                        xaxis_title="Mes", yaxis_title="Error (M$)",
                    )
                    st.plotly_chart(fig_val, width="stretch")
            else:
                st.info("Los datos de validación no están disponibles para este modelo.")

        # --- TAB 3: Metodología ---
        with t3:
            orden    = res.get("orden",    "—")
            s_orden  = res.get("seasonal_order", None)
            aic      = res.get("aic",  np.nan)
            bic      = res.get("bic",  np.nan)
            lb_p     = res.get("lb_pvalue", np.nan)

            st.markdown("**Especificación del modelo**")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Orden ARIMA",  str(orden))
            col_m1.metric("AIC",          f"{aic:.2f}" if not np.isnan(aic) else "—")
            col_m2.metric("Orden Estacional", str(s_orden) if s_orden else "Sin estacionalidad")
            col_m2.metric("BIC",          f"{bic:.2f}" if not np.isnan(bic) else "—")

            st.markdown("**Interpretación del modelo en lenguaje llano**")
            if isinstance(orden, str) and orden != "—":
                try:
                    orden_tuple = tuple(
                        int(x) for x in orden.strip("()").split(",")
                    )
                    st.markdown(interpretar_arima(orden_tuple, s_orden))
                except Exception:
                    st.markdown(f"Modelo ARIMA de orden `{orden}`.")

            if config_pretests:
                st.markdown("---")
                st.markdown("**Pre-tests aplicados**")
                res_kw    = config_pretests.get("res_kw")
                res_chow  = config_pretests.get("res_chow")
                fi        = config_pretests.get("fecha_inicio")

                if res_kw and "error" not in res_kw:
                    st.markdown(f"🔬 **Kruskal-Wallis:** {res_kw['interpretacion']}")
                if res_chow and "error" not in res_chow:
                    st.markdown(f"🔬 **Chow:** {res_chow['interpretacion']}")
                if fi:
                    st.markdown(
                        f"📅 **Período de entrenamiento:** desde {fi.strftime('%B %Y')}"
                    )

            st.markdown("---")
            st.markdown(
                "**Intervalo de confianza:** 80% (α=0.20). "
                "Las bandas representan el rango donde se espera que caiga "
                "el valor real en 8 de cada 10 períodos."
            )


# ---------------------------------------------------------------------------
# AJUSTE ARIMA
# ---------------------------------------------------------------------------

def _ajustar_y_pronosticar(
    serie: pd.Series, horizonte: int = 6,
    seasonal: bool = True, m: int = 12,
    X_hist=None, X_fut=None,
    alpha: float = 0.20,
    fecha_inicio: pd.Timestamp = None,
) -> dict:
    """
    Entrena auto_arima con la serie (opcionalmente truncada desde fecha_inicio)
    y genera pronóstico a `horizonte` meses.
    alpha=0.20 → IC 80% (bandas más ajustadas).
    Retorna dict con: forecast, conf_int, aic, bic, mape, rmse, lb_pvalue,
    orden, seasonal_order, y_test, y_pred_test, fechas_test.
    """
    from pmdarima import auto_arima
    from statsmodels.stats.diagnostic import acorr_ljungbox

    y_full = serie.dropna()

    # Aplicar recorte por ruptura estructural
    if fecha_inicio is not None and pd.Timestamp(fecha_inicio) > y_full.index.min():
        y_full = y_full[y_full.index >= fecha_inicio]

    if X_hist is not None and fecha_inicio is not None:
        X_hist_align = X_hist[y_full.index.min():]
        # Alinear exactamente
        idx_comun = y_full.index.intersection(
            pd.DatetimeIndex(
                [y_full.index.min() + pd.DateOffset(months=i) for i in range(len(y_full))]
            )
        )
        pass  # X_hist passed as np.ndarray; truncation handled outside

    n_test = min(3, len(y_full) - 4)
    if n_test <= 0:
        n_test = 1

    y_train = y_full.iloc[:-n_test].values
    y_test  = y_full.iloc[-n_test:].values
    fechas_test = y_full.iloc[-n_test:].index.tolist()

    # Modelo de validación (sin los últimos n_test)
    try:
        mod_val = auto_arima(
            y_train,
            X=X_hist[:-n_test] if X_hist is not None else None,
            seasonal=seasonal, m=m, stepwise=True,
            suppress_warnings=True, error_action="ignore", max_order=10,
        )
        pred_val = mod_val.predict(
            n_periods=n_test,
            X=X_hist[-n_test:] if X_hist is not None else None,
        )
        mape_val = calcular_mape(y_test, pred_val)
        rmse_val = calcular_rmse(y_test, pred_val)
    except Exception:
        pred_val  = [np.nan] * n_test
        mape_val  = np.nan
        rmse_val  = np.nan

    # Modelo final con toda la serie
    modelo = auto_arima(
        y_full.values, X=X_hist,
        seasonal=seasonal, m=m, stepwise=True,
        suppress_warnings=True, error_action="ignore", max_order=10,
    )

    forecast, conf_int = modelo.predict(
        n_periods=horizonte, X=X_fut,
        return_conf_int=True, alpha=alpha,
    )

    # Ljung-Box sobre residuos (lag=10)
    try:
        lb  = acorr_ljungbox(modelo.resid(), lags=[10], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = np.nan

    # Orden estacional del modelo
    try:
        s_order = modelo.seasonal_order
    except Exception:
        s_order = None

    return {
        "forecast":      forecast,
        "conf_int":      conf_int,
        "aic":           modelo.aic(),
        "bic":           modelo.bic(),
        "mape":          mape_val,
        "rmse":          rmse_val,
        "lb_pvalue":     lb_p,
        "orden":         str(modelo.order),
        "seasonal_order": s_order,
        "y_test":        y_test.tolist(),
        "y_pred_test":   [float(v) for v in pred_val],
        "fechas_test":   [str(f) for f in fechas_test],
    }


# ---------------------------------------------------------------------------
# WRAPPERS CACHEABLES
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def pronosticar_serie_cache(
    vals: tuple, fechas_iso: tuple,
    horizonte: int = 6, seasonal: bool = True, m: int = 12,
    fecha_inicio_iso: str = None,
) -> dict:
    """Pronóstico simple (sin exógenas). IC al 80%."""
    serie = pd.Series(list(vals), index=pd.to_datetime(list(fechas_iso)))
    fi    = pd.Timestamp(fecha_inicio_iso) if fecha_inicio_iso else None
    try:
        res = _ajustar_y_pronosticar(
            serie, horizonte=horizonte, seasonal=seasonal, m=m,
            alpha=0.20, fecha_inicio=fi,
        )
        return {
            "forecast":      res["forecast"].tolist(),
            "ci_lower":      res["conf_int"][:, 0].tolist(),
            "ci_upper":      res["conf_int"][:, 1].tolist(),
            "aic": res["aic"], "bic": res["bic"],
            "mape": res["mape"], "rmse": res["rmse"],
            "lb_pvalue": res["lb_pvalue"], "orden": res["orden"],
            "seasonal_order": res["seasonal_order"],
            "y_test":       res["y_test"],
            "y_pred_test":  res["y_pred_test"],
            "fechas_test":  res["fechas_test"],
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def pronosticar_macro_cache(vals: tuple, fechas_iso: tuple, horizonte: int = 6) -> dict:
    """Pronóstico de variable macro (sin estacionalidad, sin exógenas)."""
    serie = pd.Series(list(vals), index=pd.to_datetime(list(fechas_iso)))
    try:
        res = _ajustar_y_pronosticar(serie, horizonte=horizonte, seasonal=False, alpha=0.20)
        return {
            "forecast":  res["forecast"].tolist(),
            "ci_lower":  res["conf_int"][:, 0].tolist(),
            "ci_upper":  res["conf_int"][:, 1].tolist(),
            "aic": res["aic"], "bic": res["bic"], "orden": res["orden"],
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def pronosticar_con_exogenas_cache(
    y_vals: tuple, y_fechas: tuple,
    X_hist_dict: dict, X_fut_dict: dict,
    horizonte: int = 6,
    fecha_inicio_iso: str = None,
) -> dict:
    """Pronóstico ARIMAX con variables exógenas. IC al 80%."""
    serie   = pd.Series(list(y_vals), index=pd.to_datetime(list(y_fechas)))
    nombres = list(X_hist_dict.keys())
    X_hist  = np.column_stack([X_hist_dict[k] for k in nombres])
    X_fut   = np.column_stack([X_fut_dict[k]  for k in nombres])
    fi      = pd.Timestamp(fecha_inicio_iso) if fecha_inicio_iso else None
    try:
        res = _ajustar_y_pronosticar(
            serie, horizonte=horizonte, seasonal=True, m=12,
            X_hist=X_hist, X_fut=X_fut, alpha=0.20, fecha_inicio=fi,
        )
        return {
            "forecast":  res["forecast"].tolist(),
            "ci_lower":  res["conf_int"][:, 0].tolist(),
            "ci_upper":  res["conf_int"][:, 1].tolist(),
            "aic": res["aic"], "bic": res["bic"],
            "mape": res["mape"], "rmse": res["rmse"],
            "lb_pvalue": res["lb_pvalue"], "orden": res["orden"],
            "seasonal_order": res["seasonal_order"],
            "y_test":      res["y_test"],
            "y_pred_test": res["y_pred_test"],
            "fechas_test": res["fechas_test"],
        }
    except Exception as e:
        return {"error": str(e)}


def generar_fechas_futuras(ultima: pd.Timestamp, horizonte: int):
    fechas = pd.date_range(
        start=ultima + pd.DateOffset(months=1),
        periods=horizonte, freq="MS"
    )
    return [f.to_pydatetime() for f in fechas]   # 🔥 lista de datetimes nativos
