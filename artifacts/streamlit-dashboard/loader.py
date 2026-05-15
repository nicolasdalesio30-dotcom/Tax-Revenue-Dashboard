"""
loader.py — Carga y preparación de datos
=========================================
Maneja la recaudación, variables macro, REM del BCRA y proyección del IPC.
"""

import io
import pathlib
import warnings

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
EXCEL_RECA  = pathlib.Path(__file__).parent.parent.parent / (
    "attached_assets/Recaudación_2023_hasta_2026_Nominal_y_Real_1778714645756.xlsx"
)
EXCEL_MACRO = pathlib.Path(__file__).parent.parent.parent / (
    "attached_assets/Datos_macro_modelo_reca_1778720180502.xlsx"
)
REM_URL = ("https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/"
           "informes/historico-relevamiento-expectativas-mercado.xlsx")

# ---------------------------------------------------------------------------
# CONSTANTES — mapeo impuesto → fila del Excel (índice 0-based)
# ---------------------------------------------------------------------------
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

COMPONENTES_TORTA = [
    "Ganancias", "IVA", "Internos coparticipados", "Bienes personales",
    "Créditos y Débitos en cta. cte.", "Combustibles Total", "Monotributo impositivo",
    "Derechos de importación", "Derechos de exportación", "Tasa de estadística",
    "Aportes personales", "Contribuciones patronales",
]

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

# ---------------------------------------------------------------------------
# RECAUDACIÓN
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando recaudación…")
def cargar_datos() -> dict:
    """Carga el Excel de recaudación. Devuelve dict con Nominal, Real y variaciones."""
    if not EXCEL_RECA.exists():
        st.error(f"No se encontró el Excel: {EXCEL_RECA}")
        st.stop()

    raw_nom  = pd.read_excel(EXCEL_RECA, sheet_name="Nominal", header=None)
    raw_real = pd.read_excel(EXCEL_RECA, sheet_name="Real",    header=None)

    def _parsear(raw):
        fechas  = pd.to_datetime(raw.iloc[8, 2:], errors="coerce")
        mask    = fechas.notna()
        fechas  = fechas[mask]
        result  = {}
        for nombre, fila in TAX_ROW_INDEX.items():
            vals = pd.to_numeric(raw.iloc[fila, 2:][mask], errors="coerce")
            result[nombre] = pd.Series(vals.values, index=fechas)
        return pd.DataFrame(result)

    df_nom  = _parsear(raw_nom)
    df_real = _parsear(raw_real)

    return {
        "Nominal":    df_nom,
        "Real":       df_real,
        "Var Nominal": df_nom.pct_change(),
        "Var Real":    df_real.pct_change(),
        "IA Nominal":  df_nom.pct_change(12),
        "IA Real":     df_real.pct_change(12),
    }


# ---------------------------------------------------------------------------
# MACRO
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando datos macroeconómicos…")
def cargar_datos_macro() -> dict:
    """
    Carga las 4 variables macro del Excel adjunto.
    IPC: mensual. Dolar / Tasa: diario → último del mes. EMAE: mensual.
    """
    if not EXCEL_MACRO.exists():
        st.error(f"No se encontró el archivo macro: {EXCEL_MACRO}")
        st.stop()

    result = {}

    # --- IPC ---
    raw_ipc = pd.read_excel(EXCEL_MACRO, sheet_name="IPC", header=None)
    fechas_ipc = pd.to_datetime(raw_ipc.iloc[3:, 0], errors="coerce")
    vals_ipc   = pd.to_numeric(raw_ipc.iloc[3:, 1], errors="coerce")
    ipc = pd.Series(vals_ipc.values, index=fechas_ipc).dropna()
    ipc.index = pd.to_datetime(ipc.index).to_period("M").to_timestamp()
    result["IPC"] = ipc

    # --- Dólar oficial (venta) ---
    raw_d = pd.read_excel(EXCEL_MACRO, sheet_name="Dolar oficial", header=None)
    fechas_d = pd.to_datetime(raw_d.iloc[1:, 0], errors="coerce")
    vals_d   = pd.to_numeric(raw_d.iloc[1:, 2], errors="coerce")
    dolar_d  = pd.Series(vals_d.values, index=fechas_d).dropna()
    result["Dolar"] = dolar_d.resample("MS").last()

    # --- Tasa de interés ---
    raw_t = pd.read_excel(EXCEL_MACRO, sheet_name="Tasa de interes", header=None)
    fechas_t = pd.to_datetime(raw_t.iloc[1:, 0], errors="coerce")
    vals_t   = pd.to_numeric(raw_t.iloc[1:, 1], errors="coerce")
    tasa_d   = pd.Series(vals_t.values, index=fechas_t).dropna()
    result["Tasa"] = tasa_d.resample("MS").last()

    # --- EMAE ---
    raw_e = pd.read_excel(EXCEL_MACRO, sheet_name="EMAE", header=None)
    dat   = raw_e.iloc[3:, :3].copy()
    dat.columns = ["anio", "mes_str", "valor"]
    dat["anio"]    = dat["anio"].ffill()
    dat["anio"]    = pd.to_numeric(dat["anio"], errors="coerce")
    dat["valor"]   = pd.to_numeric(dat["valor"], errors="coerce")
    dat["mes_str"] = dat["mes_str"].astype(str).str.strip().str.lower()
    dat["mes_num"] = dat["mes_str"].map(MESES_ES)
    dat = dat.dropna(subset=["anio", "mes_num", "valor"])
    fechas_e = pd.to_datetime({
        "year": dat["anio"].astype(int),
        "month": dat["mes_num"].astype(int),
        "day": 1,
    })
    result["EMAE"] = pd.Series(dat["valor"].values, index=fechas_e)

    return result


# ---------------------------------------------------------------------------
# REM — Relevamiento de Expectativas de Mercado (BCRA)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Descargando REM (BCRA)…")
def cargar_rem() -> tuple:
    """
    Descarga el Excel histórico del REM del BCRA.
    Retorna:
        rem_serie: pd.Series   — inflación mensual esperada (mediana, fracción decimal)
                                  indexada por Período (timestamp del primer día del mes)
        fecha_rem: pd.Timestamp — fecha de corte del último relevamiento
    En caso de error retorna (Serie vacía, None).
    """
    try:
        resp = requests.get(REM_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content),
                           sheet_name="Base de Datos Completa", header=1)

        # Filtrar IPC mensual
        mask_ipc = (
            df["Variable"].str.contains("Precios minoristas|IPC", case=False, na=False)
            & df["Referencia"].str.contains("mensual", case=False, na=False)
        )
        df_ipc = df[mask_ipc].copy()
        df_ipc["Fecha de pronóstico"] = pd.to_datetime(df_ipc["Fecha de pronóstico"],
                                                        errors="coerce")
        df_ipc["Período"] = pd.to_datetime(df_ipc["Período"], errors="coerce")

        fecha_rem = df_ipc["Fecha de pronóstico"].max()

        # Para cada período, tomar la mediana del relevamiento más reciente
        df_ipc = df_ipc.sort_values("Fecha de pronóstico")
        rem_mensual = (
            df_ipc.groupby("Período")["Mediana"]
            .last()   # último relevamiento disponible para ese período
            .dropna()
        )
        rem_mensual.index = pd.to_datetime(rem_mensual.index).to_period("M").to_timestamp()

        # Convertir de % a fracción decimal
        rem_serie = rem_mensual / 100.0

        return rem_serie, fecha_rem

    except Exception as e:
        return pd.Series(dtype=float), None


# ---------------------------------------------------------------------------
# PROYECCIÓN DEL IPC Y EXTENSIÓN DE LA SERIE REAL
# ---------------------------------------------------------------------------

def proyectar_real_con_rem(df_nom: pd.DataFrame, df_real: pd.DataFrame,
                            ipc_serie: pd.Series, rem_serie: pd.Series) -> tuple:
    """
    Extiende df_real a los meses donde df_nom tiene datos pero df_real no.
    Usa REM para proyectar el IPC y deflactar los valores nominales.

    Retorna:
        df_real_ext: DataFrame extendido
        info_proyeccion: dict {mes: tasa_mensual_usada}  (tasas en %, no fracción)
    """
    ultimo_real = df_real.index.max()
    ultimo_nom  = df_nom.index.max()

    if ultimo_nom <= ultimo_real:
        return df_real.copy(), {}

    meses_gap = pd.date_range(
        start=ultimo_real + pd.DateOffset(months=1),
        end=ultimo_nom,
        freq="MS",
    )

    # Factor de deflación del último mes conocido (Nominal / Real para TOTAL)
    total_nom_base  = df_nom.loc[ultimo_real, "TOTAL REC. TRIBUTARIOS"]
    total_real_base = df_real.loc[ultimo_real, "TOTAL REC. TRIBUTARIOS"]
    deflactor_base  = total_nom_base / total_real_base  # IPC_t / IPC_referencia

    ipc_ultimo = ipc_serie.iloc[-1] if not ipc_serie.empty else 1.0

    df_real_ext   = df_real.copy()
    info_proyec   = {}
    ipc_proyec    = ipc_ultimo
    deflactor_act = deflactor_base

    for mes in meses_gap:
        # Buscar la tasa REM para este mes; fallback = promedio últimos 3 meses del REM
        if mes in rem_serie.index and not np.isnan(rem_serie[mes]):
            tasa = float(rem_serie[mes])
        elif not rem_serie.empty:
            tasa = float(rem_serie.iloc[-3:].mean())
        else:
            tasa = 0.03  # 3% mensual como fallback genérico

        ipc_proyec    = ipc_proyec    * (1 + tasa)
        deflactor_act = deflactor_base * (ipc_proyec / ipc_ultimo)

        # Real_t = Nominal_t / deflactor_t  (sólo si hay datos nominales)
        nueva_fila = df_nom.loc[mes] / deflactor_act
        df_real_ext.loc[mes] = nueva_fila
        # Sólo reportar si hay al menos un valor real proyectado válido
        if nueva_fila.notna().any():
            info_proyec[mes] = tasa * 100  # guardar en %

    return df_real_ext.sort_index(), info_proyec
