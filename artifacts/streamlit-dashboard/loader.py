"""
loader.py — Carga y preparación de datos
=========================================
Maneja la recaudación, variables macro, REM del BCRA y proyección del IPC.
Soporta tanto archivos en disco como archivos subidos por el usuario.
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
# RUTAS POR DEFECTO
# ---------------------------------------------------------------------------
EXCEL_RECA  = pathlib.Path(__file__).parent.parent.parent / (
    "attached_assets/Recaudación_2023_hasta_2026_Nominal_y_Real_1778714645756.xlsx"
)
EXCEL_MACRO = pathlib.Path(__file__).parent.parent.parent / (
    "attached_assets/Datos_macro_modelo_reca_1778720180502.xlsx"
)
REM_URL = (
    "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/"
    "informes/historico-relevamiento-expectativas-mercado.xlsx"
)

# ---------------------------------------------------------------------------
# MAPEOS
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

def _parsear_reca(src) -> dict:
    """Parsea el Excel de recaudación desde una ruta o BytesIO."""
    raw_nom  = pd.read_excel(src, sheet_name="Nominal", header=None)
    raw_real = pd.read_excel(src, sheet_name="Real",    header=None)

    def _df(raw):
        fechas = pd.to_datetime(raw.iloc[8, 2:], errors="coerce")
        mask   = fechas.notna()
        fechas = fechas[mask]
        result = {}
        for nombre, fila in TAX_ROW_INDEX.items():
            vals = pd.to_numeric(raw.iloc[fila, 2:][mask], errors="coerce")
            result[nombre] = pd.Series(vals.values, index=fechas)
        return pd.DataFrame(result)

    df_nom  = _df(raw_nom)
    df_real = _df(raw_real)

    # Descartar filas en que TODOS los impuestos son NaN (meses sin datos)
    df_nom  = df_nom.dropna(how="all")
    df_real = df_real.dropna(how="all")

    return {
        "Nominal":    df_nom,
        "Real":       df_real,
        "Var Nominal": df_nom.pct_change(),
        "Var Real":    df_real.pct_change(),
        "IA Nominal":  df_nom.pct_change(12),
        "IA Real":     df_real.pct_change(12),
    }


@st.cache_data(show_spinner="Cargando recaudación…")
def cargar_datos(reca_bytes: bytes = None) -> dict:
    """
    Carga el Excel de recaudación.
    Si reca_bytes no es None, los usa en lugar del archivo en disco.
    """
    if reca_bytes is not None:
        src = io.BytesIO(reca_bytes)
    else:
        if not EXCEL_RECA.exists():
            st.error(f"No se encontró el Excel: {EXCEL_RECA}")
            st.stop()
        src = EXCEL_RECA
    return _parsear_reca(src)


# ---------------------------------------------------------------------------
# MACRO
# ---------------------------------------------------------------------------

def _parsear_macro(src) -> dict:
    """Parsea el Excel macro desde una ruta o BytesIO."""
    result = {}

    def _leer_sheet(sheet):
        return pd.read_excel(src if not isinstance(src, io.BytesIO)
                             else io.BytesIO(src.getvalue() if hasattr(src, "getvalue") else src.read()),
                             sheet_name=sheet, header=None)

    # Para BytesIO necesitamos guardar el contenido
    if isinstance(src, io.BytesIO):
        raw_bytes = src.read()
        def _leer(sheet):
            return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet, header=None)
    else:
        def _leer(sheet):
            return pd.read_excel(src, sheet_name=sheet, header=None)

    # --- IPC ---
    raw_ipc = _leer("IPC")
    fechas_ipc = pd.to_datetime(raw_ipc.iloc[3:, 0], errors="coerce")
    vals_ipc   = pd.to_numeric(raw_ipc.iloc[3:, 1], errors="coerce")
    ipc = pd.Series(vals_ipc.values, index=fechas_ipc).dropna()
    ipc.index = pd.to_datetime(ipc.index).to_period("M").to_timestamp()
    result["IPC"] = ipc

    # --- Dólar oficial ---
    raw_d = _leer("Dolar oficial")
    fechas_d = pd.to_datetime(raw_d.iloc[1:, 0], errors="coerce")
    vals_d   = pd.to_numeric(raw_d.iloc[1:, 2], errors="coerce")
    dolar_d  = pd.Series(vals_d.values, index=fechas_d).dropna()
    result["Dolar"] = dolar_d.resample("MS").last()

    # --- Tasa de interés ---
    raw_t = _leer("Tasa de interes")
    fechas_t = pd.to_datetime(raw_t.iloc[1:, 0], errors="coerce")
    vals_t   = pd.to_numeric(raw_t.iloc[1:, 1], errors="coerce")
    tasa_d   = pd.Series(vals_t.values, index=fechas_t).dropna()
    result["Tasa"] = tasa_d.resample("MS").last()

    # --- EMAE ---
    raw_e = _leer("EMAE")
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


@st.cache_data(show_spinner="Cargando datos macroeconómicos…")
def cargar_datos_macro(macro_bytes: bytes = None) -> dict:
    """
    Carga el Excel macro.
    Si macro_bytes no es None, los usa en lugar del archivo en disco.
    """
    if macro_bytes is not None:
        src = io.BytesIO(macro_bytes)
    else:
        if not EXCEL_MACRO.exists():
            st.error(f"No se encontró el archivo macro: {EXCEL_MACRO}")
            st.stop()
        src = EXCEL_MACRO
    return _parsear_macro(src)


# ---------------------------------------------------------------------------
# REM — Relevamiento de Expectativas de Mercado (BCRA)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Descargando REM (BCRA)…")
def cargar_rem(rem_bytes: bytes = None) -> tuple:
    """
    Descarga el Excel histórico del REM del BCRA, o lo carga desde bytes subidos.
    Retorna:
        rem_serie   — pd.Series de inflación mensual esperada (fracción decimal)
                      indexada por primer día del mes
        fecha_rem   — pd.Timestamp del último relevamiento
        df_completo — DataFrame completo del REM para mostrar en la UI
    En caso de error retorna (Serie vacía, None, DataFrame vacío).
    """
    try:
        if rem_bytes is not None:
            raw = rem_bytes
        else:
            resp = requests.get(REM_URL, timeout=30,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.content

        df = pd.read_excel(io.BytesIO(raw),
                           sheet_name="Base de Datos Completa", header=1)

        # Filtrar IPC mensual
        mask = (
            df["Variable"].str.contains("Precios minoristas|IPC", case=False, na=False)
            & df["Referencia"].str.contains("mensual", case=False, na=False)
        )
        df_ipc = df[mask].copy()
        df_ipc["Fecha de pronóstico"] = pd.to_datetime(
            df_ipc["Fecha de pronóstico"], errors="coerce"
        )
        df_ipc["Período"] = pd.to_datetime(df_ipc["Período"], errors="coerce")

        fecha_rem = df_ipc["Fecha de pronóstico"].max()

        # Para cada período, tomar la mediana del relevamiento más reciente
        df_ipc = df_ipc.sort_values("Fecha de pronóstico")
        rem_por_periodo = (
            df_ipc.groupby("Período")
            .agg(
                Mediana=("Mediana", "last"),
                Promedio=("Promedio", "last"),
                Desvio=("Desvío", "last"),
                Fecha_pronostico=("Fecha de pronóstico", "last"),
            )
            .reset_index()
        )
        rem_por_periodo["Período"] = pd.to_datetime(
            rem_por_periodo["Período"]
        ).dt.to_period("M").dt.to_timestamp()

        rem_serie = rem_por_periodo.set_index("Período")["Mediana"] / 100.0

        # DataFrame para mostrar en UI (últimos 24 períodos)
        df_ui = rem_por_periodo.tail(24).copy()
        df_ui["Período"] = df_ui["Período"].dt.strftime("%b %Y")
        df_ui["Mediana (%)"]  = df_ui["Mediana"].round(2)
        df_ui["Promedio (%)"] = df_ui["Promedio"].round(2)
        df_ui["Desvío (%)"]   = df_ui["Desvio"].round(2)
        df_ui = df_ui[["Período", "Mediana (%)", "Promedio (%)", "Desvío (%)",
                        "Fecha_pronostico"]].rename(
            columns={"Fecha_pronostico": "Fecha del relevamiento"}
        )

        return rem_serie, fecha_rem, df_ui

    except Exception as e:
        return pd.Series(dtype=float), None, pd.DataFrame()


# ---------------------------------------------------------------------------
# PROYECCIÓN DEL IPC Y EXTENSIÓN DE LA SERIE REAL
# ---------------------------------------------------------------------------

def proyectar_real_con_rem(
    df_nom: pd.DataFrame, df_real: pd.DataFrame,
    ipc_serie: pd.Series, rem_serie: pd.Series,
) -> tuple:
    """
    Extiende df_real a los meses donde df_nom tiene datos pero df_real no.
    Usa el REM para proyectar el IPC y deflactar los valores nominales.

    Retorna:
        df_real_ext     — DataFrame extendido con los meses proyectados
        info_proyeccion — dict {mes: tasa_mensual_usada (%)}
    """
    ultimo_real = df_real.index.max()
    ultimo_nom  = df_nom.index.max()

    if ultimo_nom <= ultimo_real:
        return df_real.copy(), {}

    meses_gap = pd.date_range(
        start=ultimo_real + pd.DateOffset(months=1),
        end=ultimo_nom, freq="MS",
    )

    # Factor de deflación del último mes conocido
    total_nom_base  = df_nom.loc[ultimo_real, "TOTAL REC. TRIBUTARIOS"]
    total_real_base = df_real.loc[ultimo_real, "TOTAL REC. TRIBUTARIOS"]
    if pd.isna(total_nom_base) or pd.isna(total_real_base) or total_real_base == 0:
        return df_real.copy(), {}

    deflactor_base = total_nom_base / total_real_base
    ipc_ultimo     = ipc_serie.iloc[-1] if not ipc_serie.empty else 1.0

    df_real_ext  = df_real.copy()
    info_proyec  = {}
    ipc_proyec   = ipc_ultimo
    deflactor    = deflactor_base

    for mes in meses_gap:
        # Tasa mensual: usar REM si disponible, sino promedio últimos 3m del REM
        if mes in rem_serie.index and not np.isnan(rem_serie[mes]):
            tasa = float(rem_serie[mes])
        elif not rem_serie.empty:
            tasa = float(rem_serie.iloc[-3:].mean())
        else:
            tasa = 0.03

        ipc_proyec = ipc_proyec  * (1 + tasa)
        deflactor  = deflactor_base * (ipc_proyec / ipc_ultimo)

        nueva_fila = df_nom.loc[mes] / deflactor
        df_real_ext.loc[mes] = nueva_fila

        if nueva_fila.notna().any():
            info_proyec[mes] = tasa * 100

    return df_real_ext.sort_index(), info_proyec
