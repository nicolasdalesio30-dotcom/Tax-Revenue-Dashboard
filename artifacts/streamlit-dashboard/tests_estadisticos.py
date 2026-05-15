"""
tests_estadisticos.py — Pre-tests estadísticos para configurar Auto-ARIMA
===========================================================================
Implementa:
  - Test de Kruskal-Wallis: detectar estacionalidad mensual
  - Test de Chow: detectar ruptura estructural en una fecha dada
  - Escáner automático de ruptura: encuentra el punto de quiebre más significativo
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# KRUSKAL-WALLIS — Test de estacionalidad no paramétrico
# ---------------------------------------------------------------------------

MESES_CORTOS = {
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
    7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic",
}


def test_estacionalidad_kw(serie: pd.Series) -> dict:
    """
    Test de Kruskal-Wallis sobre los 12 grupos mensuales.

    H₀: Todos los meses tienen la misma distribución (sin estacionalidad)
    H₁: Al menos un mes difiere (hay estacionalidad)

    Recomendación: si p < 0.05 → seasonal=True, m=12
    """
    y = serie.dropna()
    grupos = []
    grupos_labels = []
    for m in range(1, 13):
        g = y[y.index.month == m].values
        if len(g) >= 2:
            grupos.append(g)
            grupos_labels.append(MESES_CORTOS[m])

    if len(grupos) < 4:
        return {"error": "Insuficientes meses con ≥2 observaciones para el test."}

    stat, p_value = sp_stats.kruskal(*grupos)
    hay_estac = p_value < 0.05

    # Medias por mes para el gráfico
    medias_por_mes = {
        MESES_CORTOS[m]: float(y[y.index.month == m].mean())
        for m in range(1, 13) if len(y[y.index.month == m]) >= 1
    }

    return {
        "stat":              stat,
        "p_value":           p_value,
        "hay_estacionalidad": hay_estac,
        "seasonal_recomendado": hay_estac,
        "m_recomendado":     12 if hay_estac else 1,
        "n_grupos":          len(grupos),
        "medias_por_mes":    medias_por_mes,
        "interpretacion": (
            f"✅ Estacionalidad detectada (H={stat:.2f}, p={p_value:.4f} < 0.05) "
            f"→ se recomienda seasonal=True, m=12"
            if hay_estac else
            f"⬜ Sin estacionalidad significativa (H={stat:.2f}, p={p_value:.4f} ≥ 0.05) "
            f"→ se recomienda seasonal=False"
        ),
    }


# ---------------------------------------------------------------------------
# CHOW — Test de ruptura estructural en una fecha dada
# ---------------------------------------------------------------------------

def test_chow(serie: pd.Series, fecha_ruptura: pd.Timestamp) -> dict:
    """
    Test de Chow para ruptura estructural en `fecha_ruptura`.

    Modelo: y = α + β·t (tendencia lineal)
    H₀: (α₁, β₁) = (α₂, β₂)  — sin cambio de parámetros
    H₁: Los parámetros difieren antes y después de la ruptura

    Incluye un error si los segmentos son muy pequeños.
    """
    try:
        from statsmodels.api import OLS, add_constant
        from scipy.stats import f as f_dist
    except ImportError:
        return {"error": "statsmodels no disponible"}

    y = serie.dropna()
    n = len(y)
    t_seq = np.arange(n, dtype=float)
    X = add_constant(t_seq)

    break_idx = int((y.index < fecha_ruptura).sum())
    obs_pre   = break_idx
    obs_post  = n - break_idx

    if obs_pre < 4 or obs_post < 4:
        return {
            "error": f"Segmentos insuficientes: {obs_pre} obs. antes, "
                     f"{obs_post} obs. después (mín. 4 cada uno)."
        }

    rss_full = OLS(y.values, X).fit().ssr
    rss1     = OLS(y.values[:break_idx], X[:break_idx]).fit().ssr
    rss2     = OLS(y.values[break_idx:], X[break_idx:]).fit().ssr

    k = X.shape[1]
    denom = (rss1 + rss2) / max(n - 2 * k, 1)
    F_stat   = ((rss_full - (rss1 + rss2)) / k) / denom
    p_value  = float(1 - f_dist.cdf(F_stat, dfn=k, dfd=max(n - 2 * k, 1)))
    hay_rupt = p_value < 0.05

    return {
        "F_stat":       F_stat,
        "p_value":      p_value,
        "break_date":   fecha_ruptura,
        "break_idx":    break_idx,
        "hay_ruptura":  hay_rupt,
        "obs_pre":      obs_pre,
        "obs_post":     obs_post,
        "fecha_inicio_recomendada": fecha_ruptura if hay_rupt else y.index.min(),
        "interpretacion": (
            f"⚠️ Ruptura estructural detectada en {fecha_ruptura.strftime('%b %Y')} "
            f"(F={F_stat:.2f}, p={p_value:.4f} < 0.05).\n"
            f"Recomendación: entrenar el modelo solo desde {fecha_ruptura.strftime('%b %Y')} "
            f"({obs_post} observaciones)."
            if hay_rupt else
            f"✅ Sin ruptura significativa en {fecha_ruptura.strftime('%b %Y')} "
            f"(F={F_stat:.2f}, p={p_value:.4f} ≥ 0.05). Usar la serie completa."
        ),
    }


# ---------------------------------------------------------------------------
# ESCÁNER DE RUPTURA AUTOMÁTICO (Quandt-Andrews simplificado)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Buscando ruptura estructural…")
def detectar_ruptura_automatica(vals: tuple, fechas_iso: tuple) -> dict:
    """
    Escanea todos los posibles puntos de ruptura (20%–80% de la serie)
    y devuelve el de mayor F-stat (menor p-value).

    Equivale al test de Quandt (1960) en su variante simplificada.
    Retorna el mejor resultado de test_chow() o un dict con 'error'.
    """
    y = pd.Series(list(vals), index=pd.to_datetime(list(fechas_iso))).dropna()
    n = len(y)
    if n < 12:
        return {"error": f"Serie muy corta ({n} obs). Mínimo 12."}

    inicio = max(4, n // 5)
    fin    = min(n - 4, 4 * n // 5)

    mejor_p   = 1.0
    mejor_res = None

    for idx in range(inicio, fin):
        bd  = y.index[idx]
        res = test_chow(y, bd)
        if "error" not in res and res["p_value"] < mejor_p:
            mejor_p   = res["p_value"]
            mejor_res = res

    if mejor_res is None:
        return {"error": "No se pudo calcular ningún test de Chow."}

    mejor_res["metodo"] = "Quandt-Andrews (escáner automático)"
    return mejor_res


# ---------------------------------------------------------------------------
# RENDER: sección de pre-tests en Streamlit
# ---------------------------------------------------------------------------

def render_pretests(serie: pd.Series, nombre: str) -> dict:
    """
    Muestra los resultados de los pre-tests estadísticos en Streamlit.
    Retorna dict con configuración recomendada para Auto-ARIMA:
      {seasonal, m, fecha_inicio}
    """
    y = serie.dropna()

    st.markdown("#### 🧪 Pre-tests Estadísticos")
    st.caption(
        "Los tests determinan automáticamente si el modelo debe incluir "
        "estacionalidad y si hay un quiebre estructural que delimite el período de entrenamiento."
    )

    col_kw, col_chow = st.columns(2)

    # --- Kruskal-Wallis ---
    with col_kw:
        st.markdown("**Test de Kruskal-Wallis (estacionalidad)**")
        st.caption("H₀: distribución idéntica en todos los meses")
        res_kw = test_estacionalidad_kw(y)

        if "error" in res_kw:
            st.warning(res_kw["error"])
            seasonal_rec, m_rec = True, 12
        else:
            color_kw = "#1a7f37" if res_kw["hay_estacionalidad"] else "#0969da"
            st.markdown(
                f'<div style="border-left:4px solid {color_kw};padding:8px 12px;'
                f'background:#f6f8fa;border-radius:4px">{res_kw["interpretacion"]}</div>',
                unsafe_allow_html=True,
            )
            mk1, mk2 = st.columns(2)
            mk1.metric("H de Kruskal-Wallis", f"{res_kw['stat']:.3f}")
            mk2.metric("p-value", f"{res_kw['p_value']:.4f}")
            seasonal_rec = res_kw["hay_estacionalidad"]
            m_rec        = res_kw["m_recomendado"]

    # --- Chow automático ---
    with col_chow:
        st.markdown("**Test de Chow (ruptura estructural)**")
        st.caption("Escáner automático: busca el quiebre más significativo")

        res_chow = detectar_ruptura_automatica(
            vals=tuple(y.values.tolist()),
            fechas_iso=tuple(y.index.strftime("%Y-%m-%d").tolist()),
        )

        fecha_inicio_rec = y.index.min()

        if "error" in res_chow:
            st.info(res_chow["error"])
        else:
            color_ch = "#d14900" if res_chow["hay_ruptura"] else "#1a7f37"
            st.markdown(
                f'<div style="border-left:4px solid {color_ch};padding:8px 12px;'
                f'background:#f6f8fa;border-radius:4px">{res_chow["interpretacion"]}</div>',
                unsafe_allow_html=True,
            )
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("F de Chow", f"{res_chow['F_stat']:.3f}")
            mc2.metric("p-value",   f"{res_chow['p_value']:.4f}")
            mc3.metric("Fecha pico", res_chow["break_date"].strftime("%b %Y"))
            fecha_inicio_rec = res_chow["fecha_inicio_recomendada"]

    # --- Override manual ---
    st.markdown("---")
    with st.expander("⚙️ Ajustar configuración manualmente"):
        col_ov1, col_ov2, col_ov3 = st.columns(3)
        with col_ov1:
            usar_seasonal = st.checkbox(
                "Estacionalidad", value=seasonal_rec,
                key=f"override_seasonal_{nombre}",
                help="Si está marcado → seasonal=True, m=12"
            )
        with col_ov2:
            fechas_disp = sorted(y.index.tolist())
            idx_default = max(0, len(fechas_disp) - len(y[y.index >= fecha_inicio_rec]))
            fecha_sel = st.selectbox(
                "Inicio de entrenamiento",
                options=fechas_disp,
                index=idx_default,
                format_func=lambda d: d.strftime("%b %Y"),
                key=f"override_fecha_{nombre}",
                help="El modelo se entrena solo con datos desde esta fecha en adelante"
            )
        with col_ov3:
            st.metric("Observaciones usadas",
                      len(y[y.index >= fecha_sel]),
                      help="Número de meses que se usarán para entrenar el modelo")

    return {
        "seasonal":      usar_seasonal,
        "m":             12 if usar_seasonal else 1,
        "fecha_inicio":  fecha_sel,
        "res_kw":        res_kw if "error" not in (res_kw or {}) else None,
        "res_chow":      res_chow if "error" not in (res_chow or {}) else None,
    }
