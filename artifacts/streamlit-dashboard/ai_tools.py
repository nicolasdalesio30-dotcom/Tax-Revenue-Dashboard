"""
ai_tools.py — Scraping web real + IA
======================================
Integra Claude (Anthropic) + búsqueda y scraping web real para:
  - Boletín Oficial: busca noticias reales, no usa conocimiento del modelo
  - Sentimiento macro: análisis basado en contenido web actual
"""

import os
import re
import time
import warnings
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

MESES_LARGO = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}

HEADERS_WEB = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-AR,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ---------------------------------------------------------------------------
# BÚSQUEDA DDG (con compatibilidad para duckduckgo_search y ddgs)
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int = 6) -> list:
    """
    Búsqueda web vía DuckDuckGo.
    Compatible con ddgs (nuevo) y duckduckgo_search (legacy).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, region="ar-es"))
    except Exception:
        return []


def _ddg_news(query: str, max_results: int = 6) -> list:
    """Búsqueda de noticias recientes vía DuckDuckGo News."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            return list(ddgs.news(query, max_results=max_results, region="ar-es"))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# SCRAPING DE PÁGINAS WEB
# ---------------------------------------------------------------------------

def _scrape_url(url: str, timeout: int = 10) -> str:
    """
    Descarga una URL y extrae el texto principal (sin tags HTML).
    Retorna el texto limpio (máx. 3000 caracteres) o cadena vacía si falla.
    """
    try:
        r = requests.get(url, headers=HEADERS_WEB, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "html" not in ct:
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        # Eliminar scripts, estilos, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "iframe", "button"]):
            tag.decompose()

        # Buscar el bloque de contenido principal
        main = (soup.find("article") or soup.find("main") or
                soup.find("div", {"id": re.compile(r"content|article|body", re.I)}) or
                soup.find("div", {"class": re.compile(r"content|article|body|nota|post", re.I)}) or
                soup.body)

        if main:
            texto = main.get_text(separator=" ", strip=True)
        else:
            texto = soup.get_text(separator=" ", strip=True)

        # Limpiar espacios múltiples
        texto = re.sub(r"\s{2,}", " ", texto).strip()
        return texto[:3500]

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# CLIENTE ANTHROPIC
# ---------------------------------------------------------------------------

def _get_claude_client():
    import anthropic
    base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL", "")
    api_key  = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY", "")
    if not base_url or not api_key:
        raise RuntimeError(
            "Variables AI_INTEGRATIONS_ANTHROPIC_BASE_URL y "
            "AI_INTEGRATIONS_ANTHROPIC_API_KEY no configuradas."
        )
    return anthropic.Anthropic(base_url=base_url, api_key=api_key)


def _llamar_claude(prompt: str, max_tokens: int = 1200) -> str:
    client = _get_claude_client()
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ---------------------------------------------------------------------------
# BOLETÍN OFICIAL — Scraping real
# ---------------------------------------------------------------------------

SITIOS_OBJETIVO = [
    "infobae.com", "ambito.com", "cronista.com", "iprofesional.com",
    "pagina12.com.ar", "telam.com.ar", "afip.gob.ar", "argentina.gob.ar",
    "infoleg.gob.ar",
]

def buscar_boletin_oficial(impuesto: str, mes: int, anio: int) -> dict:
    """
    Busca publicaciones reales del Boletín Oficial y noticias financieras
    relacionadas con el impuesto en el período indicado.

    Estrategia:
      1. DDG News: {impuesto} Argentina Boletín Oficial {mes} {anio}
      2. DDG Text: AFIP ARCA {impuesto} resolución normativa {mes} {anio}
      3. Scrapea los artículos encontrados
      4. Devuelve los artículos con su texto extraído + URLs

    Retorna dict con:
      - artículos: lista de {url, title, texto_scrapeado}
      - fecha_busqueda: cuando se realizó la búsqueda
    """
    mes_str = MESES_LARGO.get(mes, str(mes))
    periodo_str = f"{mes_str} {anio}"

    queries = [
        f'{impuesto} Argentina Boletín Oficial {periodo_str} AFIP ARCA',
        f'{impuesto} normativa tributaria Argentina {anio} resolución decreto',
        f'AFIP ARCA {impuesto} {anio} cambios',
    ]

    # --- Búsqueda noticias + texto ---
    resultados_crudos = []
    for q in queries:
        for r in _ddg_news(q, max_results=4):
            resultados_crudos.append({
                "url":   r.get("url", r.get("href", "")),
                "title": r.get("title", ""),
                "date":  r.get("date", ""),
                "body":  r.get("body", ""),
            })
        for r in _ddg_search(q, max_results=3):
            resultados_crudos.append({
                "url":   r.get("href", ""),
                "title": r.get("title", ""),
                "date":  "",
                "body":  r.get("body", ""),
            })

    # Deduplicar por URL
    vistos = set()
    articulos_unicos = []
    for r in resultados_crudos:
        url = r.get("url", "")
        if url and url not in vistos and "boletinoficial.gob.ar/resultados" not in url:
            vistos.add(url)
            articulos_unicos.append(r)

    # Scrapear los primeros N artículos
    articulos_con_texto = []
    for art in articulos_unicos[:8]:
        url = art["url"]
        texto = _scrape_url(url, timeout=8)
        if texto:
            art["texto_scrapeado"] = texto
        else:
            art["texto_scrapeado"] = art.get("body", "")[:500]
        if art["texto_scrapeado"].strip():
            articulos_con_texto.append(art)

    return {
        "articulos":      articulos_con_texto,
        "periodo":        periodo_str,
        "impuesto":       impuesto,
        "fecha_busqueda": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "n_fuentes":      len(articulos_con_texto),
    }


@st.cache_data(show_spinner=False, ttl=1800)
def resumir_boletin_con_ia(impuesto: str, mes: int, anio: int) -> dict:
    """
    Versión cacheada. Busca, scrapea y sintetiza con Claude.
    Retorna:
      - resumen: texto en bullets
      - fuentes: lista de {title, url, date}
      - alerta_fuentes: int (0 si hay fuentes, 1 si sin fuentes)
    """
    datos_web = buscar_boletin_oficial(impuesto, mes, anio)
    articulos  = datos_web["articulos"]
    periodo    = datos_web["periodo"]

    # Preparar contexto para Claude
    if articulos:
        bloques_texto = []
        for i, art in enumerate(articulos[:6], 1):
            bloque = (
                f"[Fuente {i}: {art.get('title', 'Sin título')}]\n"
                f"URL: {art.get('url','')}\n"
                f"Fecha: {art.get('date','Sin fecha')}\n"
                f"Contenido: {art['texto_scrapeado'][:800]}\n"
            )
            bloques_texto.append(bloque)
        contexto = "\n---\n".join(bloques_texto)
        hay_fuentes = True
    else:
        contexto = (
            "No se encontraron artículos específicos en la búsqueda web "
            f"para {impuesto} en {periodo}."
        )
        hay_fuentes = False

    prompt = f"""Sos un analista tributario argentino. Debés resumir las novedades normativas y económicas del impuesto "{impuesto}" durante {periodo} en Argentina.

DATOS REALES obtenidos por scraping web (no uses tu conocimiento previo, solo lo que está aquí abajo):
{'='*60}
{contexto}
{'='*60}

Instrucciones ESTRICTAS:
1. Basate SOLO en el contenido de las fuentes provistas arriba. NO inventes ni completes con conocimiento propio.
2. Si la información es limitada o no hay noticias relevantes para ese impuesto/período exacto, indicalo claramente.
3. Respondé en 5 a 8 bullets en español claro y accesible.
4. Si hay resoluciones, decretos o cambios normativos mencionados, citá el número y fecha.
5. Al final, incluí: "Fuentes: [cantidad] artículos web del [fecha_búsqueda]."
6. NO incluyas introducción ni conclusión — solo bullets y la línea de fuentes.

Formato: ▸ [descripción en lenguaje simple]
"""

    try:
        resumen = _llamar_claude(prompt, max_tokens=900)
    except Exception as e:
        resumen = f"⚠️ Error al generar resumen con IA: {e}"

    fuentes = [
        {"title": a.get("title", "Sin título"), "url": a.get("url", ""),
         "date": a.get("date", "")}
        for a in articulos[:6]
    ]

    return {
        "resumen":        resumen,
        "fuentes":        fuentes,
        "hay_fuentes":    hay_fuentes,
        "periodo":        periodo,
        "fecha_busqueda": datos_web["fecha_busqueda"],
        "n_fuentes":      datos_web["n_fuentes"],
    }


# ---------------------------------------------------------------------------
# SENTIMIENTO DE MERCADO — Variables macro
# ---------------------------------------------------------------------------

DESCRIPCION_MACRO = {
    "IPC":   "inflación mensual (IPC)",
    "Dolar": "tipo de cambio dólar oficial",
    "Tasa":  "tasa de interés (depósitos 30 días)",
    "EMAE":  "actividad económica (EMAE)",
}


@st.cache_data(show_spinner=False, ttl=3600)
def analizar_sentimiento_macro(variable: str, mes: int, anio: int,
                                ultimo_valor: float, variacion_pct: float) -> dict:
    """
    Busca noticias reales sobre la variable macro y usa Claude para
    generar análisis de sentimiento de mercado.

    Retorna dict con: resumen (texto), fuentes (lista), fecha_busqueda.
    """
    mes_str  = MESES_LARGO.get(mes, str(mes))
    desc_var = DESCRIPCION_MACRO.get(variable, variable)
    periodo  = f"{mes_str} {anio}"

    queries = [
        f"Argentina {desc_var} {periodo} expectativas mercado",
        f"Argentina {variable} {anio} analistas perspectivas",
    ]

    resultados = []
    for q in queries:
        resultados.extend(_ddg_news(q, max_results=3))

    # Scraping básico
    contexto_parts = []
    fuentes = []
    for r in resultados[:5]:
        url   = r.get("url", r.get("href", ""))
        title = r.get("title", "")
        body  = r.get("body", "")
        texto = _scrape_url(url, timeout=6) if url else ""
        contenido = texto[:600] if texto else body[:300]
        if contenido.strip():
            contexto_parts.append(f"• {title}: {contenido}")
            fuentes.append({"title": title, "url": url, "date": r.get("date", "")})

    contexto = "\n".join(contexto_parts) if contexto_parts else "Sin noticias encontradas."
    var_str  = f"{variacion_pct:+.1f}%" if not pd.isna(variacion_pct) else "sin dato"

    prompt = f"""Analizá el sentimiento de mercado para la variable "{desc_var}" en Argentina durante {periodo}.

Dato observado: {ultimo_valor:.2f} | Variación mensual: {var_str}

Contenido web encontrado:
{contexto}

Respondé en exactamente 3 bullets breves en español:
▸ [Estado/nivel en ese período — basado en las noticias]
▸ [Factores que lo impulsaban — según fuentes]
▸ [Sentimiento general del mercado — optimista/neutro/pesimista y por qué]

Solo los 3 bullets, sin introducción ni cierre. Lenguaje accesible.
"""

    try:
        resumen = _llamar_claude(prompt, max_tokens=350)
    except Exception as e:
        resumen = f"⚠️ Error IA: {e}"

    return {
        "resumen":        resumen,
        "fuentes":        fuentes,
        "fecha_busqueda": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
