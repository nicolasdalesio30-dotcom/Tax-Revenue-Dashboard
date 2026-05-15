"""
ai_tools.py — Inteligencia Artificial y búsqueda web
======================================================
Integra Claude (Anthropic) y DuckDuckGo para:
  - Resumen del Boletín Oficial para el impuesto/período seleccionado
  - Análisis de sentimiento de mercado para variables macro argentinas
"""

import os
import json
import warnings
from datetime import datetime

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

MESES_LARGO = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


# ---------------------------------------------------------------------------
# CLIENTE ANTHROPIC
# ---------------------------------------------------------------------------

def _get_client():
    """Inicializa el cliente Anthropic con las env vars de Replit AI Integrations."""
    import anthropic
    base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL", "")
    api_key  = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY", "")
    if not base_url or not api_key:
        raise RuntimeError("Variables de entorno de Anthropic no configuradas.")
    return anthropic.Anthropic(base_url=base_url, api_key=api_key)


def _llamar_claude(prompt: str, max_tokens: int = 1200) -> str:
    """Realiza una llamada a Claude Haiku y devuelve el texto de la respuesta."""
    client = _get_client()
    mensaje = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return mensaje.content[0].text.strip()


# ---------------------------------------------------------------------------
# BÚSQUEDA DDG
# ---------------------------------------------------------------------------

def _buscar_noticias(query: str, max_results: int = 5) -> list:
    """
    Busca noticias usando DuckDuckGo (sin autenticación).
    Retorna lista de dicts {title, body, href}.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            resultados = list(ddgs.text(query, max_results=max_results, region="ar-es"))
        return resultados
    except Exception:
        return []


# ---------------------------------------------------------------------------
# BOLETÍN OFICIAL
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def resumir_boletin_oficial(impuesto: str, mes: int, anio: int) -> str:
    """
    Busca en el Boletín Oficial (BORA) y sitios de noticias financieras argentinos
    publicaciones relacionadas con el impuesto en el período indicado.
    Usa Claude para sintetizar en lenguaje llano (bullets).
    """
    mes_str = MESES_LARGO.get(mes, str(mes))
    periodo_str = f"{mes_str} {anio}"

    # --- Búsqueda de noticias ---
    queries = [
        f'Boletín Oficial Argentina {impuesto} {periodo_str} resolución decreto',
        f'{impuesto} Argentina {periodo_str} AFIP ARCA cambios normativa',
        f'{impuesto} Argentina {periodo_str} sitio:infobae.com OR sitio:ambito.com OR sitio:cronista.com',
    ]
    todos_resultados = []
    for q in queries:
        todos_resultados.extend(_buscar_noticias(q, max_results=3))

    # Deduplicar y tomar los primeros 8
    vistos = set()
    contexto_noticias = []
    for r in todos_resultados:
        url = r.get("href", "")
        if url and url not in vistos:
            vistos.add(url)
            contexto_noticias.append(
                f"• {r.get('title','Sin título')}: {r.get('body','')[:200]}"
            )
        if len(contexto_noticias) >= 8:
            break

    contexto_str = "\n".join(contexto_noticias) if contexto_noticias else (
        "No se encontraron noticias recientes en la búsqueda web."
    )

    # --- Prompt para Claude ---
    prompt = f"""Sos un analista tributario argentino experto. Tu tarea es resumir los principales eventos normativos y económicos relacionados con el impuesto "{impuesto}" durante el período {periodo_str} en Argentina.

Contexto de noticias y búsquedas web (puede estar incompleto):
{contexto_str}

Instrucciones:
1. Escribí un resumen en bullets (5 a 8 puntos) en español claro y accesible, sin tecnicismos innecesarios.
2. Si hay cambios normativos (resoluciones AFIP/ARCA, decretos del PEN, leyes), mencionalos con lenguaje simple.
3. Si no encontrás información específica del período, mencioná el contexto general de ese impuesto en Argentina durante esa época.
4. Cerrá con una línea de advertencia: "Verificar en boletinoficial.gob.ar para normas con fuerza legal."
5. Respondé SOLO con los bullets y la advertencia final. No agregues introducción ni conclusión.

Formato de cada bullet: "▸ [descripción en lenguaje simple]"
"""

    try:
        return _llamar_claude(prompt, max_tokens=900)
    except Exception as e:
        return f"⚠️ No se pudo generar el resumen: {e}"


# ---------------------------------------------------------------------------
# SENTIMIENTO DE MERCADO — VARIABLES MACRO
# ---------------------------------------------------------------------------

DESCRIPCION_MACRO = {
    "IPC":   "inflación mensual (IPC)",
    "Dolar": "tipo de cambio (dólar oficial)",
    "Tasa":  "tasa de interés de depósitos a 30 días",
    "EMAE":  "actividad económica (EMAE)",
}

@st.cache_data(show_spinner=False, ttl=3600)
def analizar_sentimiento_macro(variable: str, mes: int, anio: int,
                                ultimo_valor: float, variacion_pct: float) -> str:
    """
    Genera un análisis breve de sentimiento de mercado para una variable macro argentina.
    """
    mes_str   = MESES_LARGO.get(mes, str(mes))
    desc_var  = DESCRIPCION_MACRO.get(variable, variable)
    var_str   = f"{variacion_pct:+.1f}%" if not pd.isna(variacion_pct) else "sin dato"

    # Búsqueda contextual
    query = f"Argentina {desc_var} {mes_str} {anio} perspectivas analistas mercado"
    noticias = _buscar_noticias(query, max_results=4)
    contexto = "\n".join(
        f"• {r.get('title','')}: {r.get('body','')[:150]}" for r in noticias
    ) or "Sin resultados en búsqueda web."

    prompt = f"""Sos un analista macroeconómico de Argentina. Analizá el sentimiento de mercado para la variable "{desc_var}" en el contexto argentino de {mes_str} {anio}.

Dato observado: {ultimo_valor:.2f} | Variación mensual: {var_str}

Noticias y contexto:
{contexto}

Respondé en 3 bullets cortos en español:
▸ [tendencia/nivel] — ¿cómo estaba este indicador en ese período?
▸ [drivers] — ¿qué factores lo impulsaban?
▸ [sentimiento de mercado] — ¿era optimista, neutro o pesimista respecto a este indicador?

Solo los 3 bullets, sin introducción ni cierre. Lenguaje claro para analistas no economistas.
"""

    try:
        return _llamar_claude(prompt, max_tokens=400)
    except Exception as e:
        return f"⚠️ No se pudo generar análisis: {e}"
