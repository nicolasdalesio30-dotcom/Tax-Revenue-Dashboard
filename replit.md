# Dashboard de Recaudación Tributaria

Dashboard interactivo en Streamlit para visualizar y analizar la recaudación tributaria argentina. Herramienta permanente de uso interno de la oficina.

## Run & Operate

- `streamlit run artifacts/streamlit-dashboard/app.py --server.port 8000` — iniciar el dashboard
- El workflow `Dashboard Recaudación` inicia automáticamente en el puerto 8000
- Para actualizar datos: subir Excel desde el panel lateral → "Aplicar archivos nuevos" → la app limpia caché y recarga

## Stack

- Python 3.11 + Streamlit 1.57
- Pandas 2.x (`.ffill().bfill()` SIEMPRE, jamás `fillna(method=...)`)
- Plotly para gráficos (meses en español vía `eje_x_espanol(fig, fechas)` con `tickvals`/`ticktext`)
- pmdarima + statsmodels para Auto-ARIMA / ARIMAX
- scipy para tests estadísticos (Kruskal-Wallis)
- anthropic (Replit AI Integration) para BOA y sentimiento macro
- ddgs (ex duckduckgo-search) para búsqueda de noticias
- requests + BeautifulSoup4 para scraping real de artículos
- OpenPyXL para lectura del Excel

## Where things live

- `artifacts/streamlit-dashboard/app.py` — orquestador principal (6 tabs)
- `artifacts/streamlit-dashboard/loader.py` — carga recaudación, macro, REM BCRA, proyección Real
- `artifacts/streamlit-dashboard/forecast.py` — ARIMA, métricas, scoring, transparencia del modelo
- `artifacts/streamlit-dashboard/tests_estadisticos.py` — Kruskal-Wallis + Chow + render_pretests()
- `artifacts/streamlit-dashboard/ai_tools.py` — scraping web real + Claude (BOA, sentimiento)
- `artifacts/streamlit-dashboard/.streamlit/config.toml` — configuración del servidor Streamlit
- `attached_assets/Recaudación_2023_hasta_2026_Nominal_y_Real_1778714645756.xlsx` — datos tributarios
- `attached_assets/Datos_macro_modelo_reca_1778720180502.xlsx` — macro (IPC, EMAE, Dolar, Tasa)

## Secciones del Dashboard (6 pestañas)

1. **Último Mes**: KPIs + inflación mensual real (INDEC), torta de composición, tabla resumen
2. **Análisis Histórico**: serie temporal + MM3/MM6, variación mensual e interanual, aviso de proyección REM
3. **Pronóstico Simple**: Auto-ARIMA con pre-tests (KW + Chow), IC 80%, backend de transparencia, tabla REM
4. **Pronóstico Macro**: ARIMA por variable (IPC/Dolar/Tasa/EMAE), sentimiento scrapeado + Claude
5. **Modelo con Macro**: ARIMAX completo, pre-tests, heatmap correlaciones rezagadas, métricas comparativas
6. **Boletín Oficial**: scraping real de noticias (DDG + BeautifulSoup) + Claude sintetiza; NO usa conocimiento de entrenamiento

## Pre-tests Estadísticos (tests_estadisticos.py)

- **Kruskal-Wallis**: H₀ distribución idéntica en todos los meses → detecta estacionalidad mensual
  - p < 0.05: seasonal=True, m=12 | p ≥ 0.05: seasonal=False
- **Chow (escáner automático / Quandt-Andrews simplificado)**: busca el punto de quiebre más significativo (20%–80% de la serie)
  - p < 0.05: se recomienda entrenar solo desde la fecha de ruptura
- Override manual disponible para fecha de inicio y seasonal

## Carga de Archivos (File Upload)

- Panel lateral → "📤 Subir nuevos archivos"
- Acepta Excel de recaudación, macro y REM por separado
- Detección automática de cambios por hash MD5
- Al aplicar: `st.cache_data.clear()` + `st.rerun()` — datos actualizados en el acto
- Fuente activa visible: "💾 Disco" o "📤 Subido"

## Backend / Transparencia del Modelo

Expandible "🔬 Datos utilizados y metodología" — 3 tabs:
1. **Datos de entrenamiento**: tabla fechas + valores + variación mensual, indica desde qué mes
2. **Validación**: real vs. pronosticado (últimos 3 meses fuera de muestra), error absoluto + %
3. **Metodología**: orden ARIMA, interpretación en lenguaje llano, resultados de pre-tests

## Boletín Oficial — Estrategia de Scraping Real

BOA es un SPA sin API pública → estrategia alternativa:
1. DDG News Search: `{impuesto} Argentina Boletín Oficial {mes} {anio} AFIP ARCA`
2. DDG Text Search: AFIP + normativa
3. Scraping con requests + BeautifulSoup de artículos reales (Infobae, Ámbito, Cronista, etc.)
4. Claude sintetiza SOLO el contenido scrapeado (no usa knowledge cutoff)
5. Se muestran las fuentes con URLs y fechas de publicación

## Integración REM (BCRA)

- URL: `https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/informes/historico-relevamiento-expectativas-mercado.xlsx`
- Sheet "Base de Datos Completa" — filtro: "Precios minoristas|IPC" + "mensual", mediana del último relevamiento por período
- Se usa para: proyectar la serie Real en meses sin IPC (deflactar la nominal con inflación proyectada)
- Tabla de expectativas visible en "Pronóstico Simple" (expandible)

## Modelo de Scoring (0–100)

- MAPE (40%): <2%=95, <5%=82, <10%=65, <20%=45, <35%=25, >35%=8
- Ljung-Box (35%): p>0.20=95, >0.10=80, >0.05=65, >0.01=40, else=12
- CV-RMSE (25%): RMSE/media — <5%=95, <10%=82, <20%=65, <35%=45, else=20
- Etiquetas: 80-100 Excelente | 65-79 Bueno | 50-64 Aceptable | 35-49 Débil | 0-34 Revisar

## Variables de Entorno

- `AI_INTEGRATIONS_ANTHROPIC_BASE_URL` — provisto por Replit AI Integration
- `AI_INTEGRATIONS_ANTHROPIC_API_KEY` — provisto por Replit AI Integration

## Gotchas

- Caching: `@st.cache_data` con `reca_bytes`/`macro_bytes`/`rem_bytes` como parámetros → hash distinto = cache miss automático
- El CI del pronóstico es al **80%** (alpha=0.20) — deliberadamente más ajustado que el estándar 95%
- Meses en español: `eje_x_espanol(fig, fechas)` con tickvals/ticktext — NUNCA usar tickformat de Plotly
- `width="stretch"` en `st.plotly_chart()` y `st.dataframe()` (use_container_width deprecado)
- pandas 2.x: `.ffill().bfill()` siempre, jamás `fillna(method=...)`
- ddgs puede emitir RuntimeWarning por rename; suprimido en ai_tools.py con `warnings.filterwarnings`
- `detectar_ruptura_automatica()` está cacheada con `@st.cache_data` — recibe `tuple` de vals y fechas ISO
