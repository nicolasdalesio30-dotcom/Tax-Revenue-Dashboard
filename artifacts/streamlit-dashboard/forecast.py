"""
forecast.py — Modelos ARIMA, métricas y scoring
================================================
Toda la lógica econométrica: ajuste de modelos, cálculo de errores,
benchmarks visuales y puntuación unificada del modelo.
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


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
    return float(np.sqrt(np.mean((real[mask] - pred[mask]) ** 2)))


def calcular_score_modelo(mape: float, rmse: float, lb_pvalue: float,
                           media_y: float) -> int:
    """
    Puntaje unificado del modelo (0–100) a partir de tres dimensiones:
      - MAPE (40%): precisión relativa del pronóstico
      - Ljung-Box (35%): aleatoriedad de los residuos (ruido blanco)
      - CV-RMSE (25%): error relativo a la magnitud de la variable

    Retorna entero 0-100.
    """
    def _puntaje_mape(v):
        if np.isnan(v): return 50
        if v < 2:  return 95
        if v < 5:  return 82
        if v < 10: return 65
        if v < 20: return 45
        if v < 35: return 25
        return 8

    def _puntaje_lb(p):
        if np.isnan(p): return 50
        if p > 0.20: return 95
        if p > 0.10: return 80
        if p > 0.05: return 65
        if p > 0.01: return 40
        return 12

    def _puntaje_cv(r, m):
        if np.isnan(r) or np.isnan(m) or m == 0: return 50
        cv = r / abs(m) * 100
        if cv < 5:  return 95
        if cv < 10: return 82
        if cv < 20: return 65
        if cv < 35: return 45
        return 20

    p_mape = _puntaje_mape(mape)
    p_lb   = _puntaje_lb(lb_pvalue)
    p_cv   = _puntaje_cv(rmse, media_y)

    return round(0.40 * p_mape + 0.35 * p_lb + 0.25 * p_cv)


def etiqueta_score(score: int) -> tuple:
    """Devuelve (etiqueta, color_hex) según el puntaje."""
    if score >= 80: return "Excelente", "#1a7f37"
    if score >= 65: return "Bueno",     "#0969da"
    if score >= 50: return "Aceptable", "#bf8700"
    if score >= 35: return "Débil",     "#d14900"
    return "Revisar",  "#cf222e"


# ---------------------------------------------------------------------------
# BENCHMARK VISUAL DE MÉTRICAS
# ---------------------------------------------------------------------------

def _badge(texto: str, color: str) -> str:
    return (f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:0.85em;font-weight:600">{texto}</span>')


def _calidad_mape(v: float) -> tuple:
    if np.isnan(v): return "Sin datos", "#8b949e"
    if v < 2:  return "Excelente (<2%)",  "#1a7f37"
    if v < 5:  return "Bueno (<5%)",      "#0969da"
    if v < 10: return "Aceptable (<10%)", "#bf8700"
    if v < 20: return "Débil (<20%)",     "#d14900"
    return "Bajo (>20%)", "#cf222e"


def _calidad_ljung(p: float) -> tuple:
    if np.isnan(p): return "Sin datos", "#8b949e"
    if p > 0.20: return "Excelente (p>0.20)", "#1a7f37"
    if p > 0.10: return "Bueno (p>0.10)",     "#0969da"
    if p > 0.05: return "Aceptable (p>0.05)", "#bf8700"
    if p > 0.01: return "Débil (p>0.01)",     "#d14900"
    return "Autocorrelación detectada", "#cf222e"


def _calidad_cv(rmse: float, media: float) -> tuple:
    if np.isnan(rmse) or media == 0: return "Sin datos", "#8b949e"
    cv = rmse / abs(media) * 100
    if cv < 5:  return f"Excelente ({cv:.1f}%)", "#1a7f37"
    if cv < 10: return f"Bueno ({cv:.1f}%)",     "#0969da"
    if cv < 20: return f"Aceptable ({cv:.1f}%)", "#bf8700"
    if cv < 35: return f"Débil ({cv:.1f}%)",     "#d14900"
    return f"Bajo ({cv:.1f}%)", "#cf222e"


def render_metricas_completas(res: dict, media_y: float, titulo: str = "") -> None:
    """
    Muestra las métricas del modelo con:
    - Valores numéricos
    - Badge de calidad con color
    - Puntaje unificado con interpretación
    - Acordeón explicativo
    """
    mape   = res.get("mape",      np.nan)
    rmse   = res.get("rmse",      np.nan)
    lb_p   = res.get("lb_pvalue", np.nan)
    aic    = res.get("aic",       np.nan)
    bic    = res.get("bic",       np.nan)
    orden  = res.get("orden",     "—")

    score              = calcular_score_modelo(mape, rmse, lb_p, media_y)
    etiq_score, c_score = etiqueta_score(score)

    if titulo:
        st.markdown(f"**{titulo}**")

    # --- Fila 1: métricas numéricas ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAPE",  f"{mape:.1f}%"  if not np.isnan(mape) else "—",
              help="Error porcentual medio absoluto (últimos 3 meses como test)")
    c2.metric("RMSE",  f"{rmse:,.0f}"  if not np.isnan(rmse) else "—",
              help="Raíz del error cuadrático medio (mismas unidades: M$)")
    c3.metric("AIC",   f"{aic:.1f}"    if not np.isnan(aic)  else "—",
              help="Criterio de Akaike — menor es mejor (solo comparar entre modelos)")
    c4.metric("BIC",   f"{bic:.1f}"    if not np.isnan(bic)  else "—",
              help="Criterio Bayesiano — penaliza más la complejidad")
    c5.metric("Ljung-Box p",
              f"{lb_p:.3f}" if not np.isnan(lb_p) else "—",
              help="p>0.05 → residuos sin autocorrelación (ruido blanco)")

    # --- Fila 2: badges de calidad ---
    b1, b2, b3, b4 = st.columns(4)
    etq_mape, c_mape   = _calidad_mape(mape)
    etq_lb,   c_lb     = _calidad_ljung(lb_p)
    etq_cv,   c_cv     = _calidad_cv(rmse, media_y)

    with b1:
        st.markdown("**MAPE**")
        st.markdown(_badge(etq_mape, c_mape), unsafe_allow_html=True)
    with b2:
        st.markdown("**Ljung-Box**")
        st.markdown(_badge(etq_lb, c_lb), unsafe_allow_html=True)
    with b3:
        st.markdown("**RMSE / Media**")
        st.markdown(_badge(etq_cv, c_cv), unsafe_allow_html=True)
    with b4:
        st.markdown("**Puntaje del modelo**")
        st.markdown(
            f'<div style="font-size:2em;font-weight:700;color:{c_score}">{score}/100</div>'
            f'<div>{_badge(etiq_score, c_score)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"<small>Orden ARIMA: <code>{orden}</code></small>", unsafe_allow_html=True)

    with st.expander("📖 ¿Cómo interpretar estas métricas?"):
        st.markdown("""
| Métrica | Descripción | Rango ideal |
|---|---|---|
| **MAPE** | Error porcentual medio. Qué tan lejos está el pronóstico del valor real en %. | <5% excelente, <10% aceptable |
| **RMSE** | Error en millones de pesos. Compará con el valor típico de la variable. | Depende de la escala |
| **RMSE / Media** | El RMSE como % del valor promedio. Permite comparar entre impuestos. | <10% bueno |
| **AIC / BIC** | Penalizan modelos complejos. Solo son útiles para comparar modelos entre sí. | Menor es mejor |
| **Ljung-Box** | Prueba si los residuos son "ruido blanco" (el modelo capturó toda la señal). | p > 0.05 |
| **Puntaje** | Índice 0–100: MAPE (40%) + Ljung-Box (35%) + RMSE/Media (25%). | ≥65 confiable |

**Rangos del puntaje:**
- 🟢 80–100: Excelente — usar el pronóstico con confianza
- 🔵 65–79: Bueno — pronóstico confiable con reservas menores
- 🟡 50–64: Aceptable — interpretar con cautela
- 🟠 35–49: Débil — serie muy volátil o datos insuficientes
- 🔴 0–34: Revisar — ajustar el modelo o extender la historia
        """)


# ---------------------------------------------------------------------------
# AJUSTE ARIMA — función interna
# ---------------------------------------------------------------------------

def _ajustar_y_pronosticar(serie: pd.Series, horizonte: int = 6,
                            seasonal: bool = True, m: int = 12,
                            X_hist=None, X_fut=None,
                            alpha: float = 0.20) -> dict:
    """
    Entrena auto_arima y genera pronóstico.
    alpha=0.20 → intervalo de confianza al 80% (más conservador que 90%).
    Retorna dict con: forecast, conf_int, aic, bic, mape, rmse, lb_pvalue, orden.
    """
    from pmdarima import auto_arima
    from statsmodels.stats.diagnostic import acorr_ljungbox

    n_test = 3
    y_train_val = serie.iloc[:-n_test].values
    y_test_val  = serie.iloc[-n_test:].values

    # Validación in-sample (últimos 3 meses)
    try:
        mod_val = auto_arima(
            y_train_val,
            X=X_hist[:-n_test] if X_hist is not None else None,
            seasonal=seasonal, m=m, stepwise=True,
            suppress_warnings=True, error_action="ignore", max_order=10,
        )
        pred_val = mod_val.predict(
            n_periods=n_test,
            X=X_hist[-n_test:] if X_hist is not None else None,
        )
        mape_val = calcular_mape(y_test_val, pred_val)
        rmse_val = calcular_rmse(y_test_val, pred_val)
    except Exception:
        mape_val, rmse_val = np.nan, np.nan

    # Modelo final (toda la serie)
    modelo = auto_arima(
        serie.values, X=X_hist, seasonal=seasonal, m=m,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", max_order=10,
    )

    forecast, conf_int = modelo.predict(
        n_periods=horizonte, X=X_fut,
        return_conf_int=True, alpha=alpha,
    )

    # Test de Ljung-Box sobre residuos (lag=10)
    try:
        lb  = acorr_ljungbox(modelo.resid(), lags=[10], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        lb_p = np.nan

    return {
        "forecast":  forecast,
        "conf_int":  conf_int,
        "aic":       modelo.aic(),
        "bic":       modelo.bic(),
        "mape":      mape_val,
        "rmse":      rmse_val,
        "lb_pvalue": lb_p,
        "orden":     str(modelo.order),
    }


# ---------------------------------------------------------------------------
# WRAPPERS CACHEABLES
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def pronosticar_serie_cache(vals: tuple, fechas_iso: tuple,
                             horizonte: int = 6, seasonal: bool = True,
                             m: int = 12) -> dict:
    """Pronóstico simple (sin exógenas). CI al 80%."""
    serie = pd.Series(list(vals), index=pd.to_datetime(list(fechas_iso)))
    try:
        res = _ajustar_y_pronosticar(serie, horizonte=horizonte,
                                      seasonal=seasonal, m=m, alpha=0.20)
        return {
            "forecast":  res["forecast"].tolist(),
            "ci_lower":  res["conf_int"][:, 0].tolist(),
            "ci_upper":  res["conf_int"][:, 1].tolist(),
            "aic": res["aic"], "bic": res["bic"],
            "mape": res["mape"], "rmse": res["rmse"],
            "lb_pvalue": res["lb_pvalue"], "orden": res["orden"],
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def pronosticar_macro_cache(vals: tuple, fechas_iso: tuple,
                             horizonte: int = 6) -> dict:
    """Pronóstico de variable macro (sin estacionalidad, sin exógenas)."""
    serie = pd.Series(list(vals), index=pd.to_datetime(list(fechas_iso)))
    try:
        res = _ajustar_y_pronosticar(serie, horizonte=horizonte,
                                      seasonal=False, alpha=0.20)
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
) -> dict:
    """Pronóstico ARIMA con variables exógenas (ARIMAX). CI al 80%."""
    serie = pd.Series(list(y_vals), index=pd.to_datetime(list(y_fechas)))
    nombres = list(X_hist_dict.keys())
    X_hist  = np.column_stack([X_hist_dict[k] for k in nombres])
    X_fut   = np.column_stack([X_fut_dict[k]  for k in nombres])
    try:
        res = _ajustar_y_pronosticar(
            serie, horizonte=horizonte, seasonal=True, m=12,
            X_hist=X_hist, X_fut=X_fut, alpha=0.20,
        )
        return {
            "forecast":  res["forecast"].tolist(),
            "ci_lower":  res["conf_int"][:, 0].tolist(),
            "ci_upper":  res["conf_int"][:, 1].tolist(),
            "aic": res["aic"], "bic": res["bic"],
            "mape": res["mape"], "rmse": res["rmse"],
            "lb_pvalue": res["lb_pvalue"], "orden": res["orden"],
        }
    except Exception as e:
        return {"error": str(e)}


def generar_fechas_futuras(ultima: pd.Timestamp, horizonte: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=ultima + pd.DateOffset(months=1),
        periods=horizonte, freq="MS",
    )
