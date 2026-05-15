# Dashboard de Recaudación Tributaria

Dashboard interactivo en Streamlit para visualizar y analizar la recaudación tributaria argentina.

## Run & Operate

- `streamlit run artifacts/streamlit-dashboard/app.py --server.port 8000` — iniciar el dashboard
- El workflow `Dashboard Recaudación` inicia automáticamente el dashboard en el puerto 8000

## Stack

- Python 3.11 + Streamlit 1.57
- Pandas 2.x para procesamiento de datos (usar `.ffill().bfill()`, NO `fillna(method=...)`)
- Plotly para gráficos interactivos (meses en español vía `tickvals`/`ticktext`)
- pmdarima + statsmodels para Auto-ARIMA / ARIMAX
- anthropic (Replit AI Integration) para resumen Boletín Oficial y sentimiento macro
- duckduckgo-search para búsqueda de noticias en Python
- OpenPyXL para lectura del Excel

## Where things live

- `artifacts/streamlit-dashboard/app.py` — aplicación principal (orquestador de tabs)
- `artifacts/streamlit-dashboard/loader.py` — carga de datos: recaudación, macro, REM BCRA, proyección Real
- `artifacts/streamlit-dashboard/forecast.py` — modelos ARIMA, métricas, scoring y benchmarks visuales
- `artifacts/streamlit-dashboard/ai_tools.py` — Claude + DDG: resumen BOA y sentimiento macro
- `artifacts/streamlit-dashboard/.streamlit/config.toml` — configuración del servidor Streamlit
- `attached_assets/Recaudación_2023_hasta_2026_Nominal_y_Real_1778714645756.xlsx` — fuente de datos tributarios
- `attached_assets/Datos_macro_modelo_reca_1778720180502.xlsx` — macro (IPC, EMAE, Dolar, Tasa)

## Estructura del Excel de Recaudación

- **Nominal**: recaudación en valores corrientes
- **Real**: recaudación ajustada por inflación (base 2023)
- **Variación Nominal / Real**: calculadas dinámicamente desde los datos base
- Fila 9 (Excel): fechas (columna C en adelante, mensual desde ene-2023)
- Columna B: nombres de los impuestos

## Secciones del Dashboard (6 pestañas)

1. **Último Mes**: KPIs principales + inflación mensual ("¿Cuánto dio la inflación?"), torta de composición, tabla resumen
2. **Análisis Histórico**: serie temporal con medias móviles 3M y 6M, variaciones mensual e interanual, aviso de inflación proyectada si hay meses sin IPC
3. **Pronóstico Simple**: Auto-ARIMA estacional, IC al 80%, métricas con benchmarks y puntaje unificado 0-100
4. **Pronóstico Macro**: Proyecciones individuales de IPC, Dólar, Tasa y EMAE + análisis de sentimiento de mercado con IA
5. **Modelo con Macro**: ARIMAX (recaudación + variables macro), heatmap de correlaciones por rezago (0-4 meses), comparativa de métricas
6. **Boletín Oficial**: Resumen de publicaciones del BOA para el impuesto y período seleccionados, generado con Claude + búsqueda web

## Integración REM (BCRA)

- URL del Excel histórico: `https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/informes/historico-relevamiento-expectativas-mercado.xlsx`
- Sheet: "Base de Datos Completa" — columnas: Fecha de pronóstico, Variable, Referencia, Período, Mediana
- Filtra "Precios minoristas|IPC" + "var. % mensual", toma la mediana del último relevamiento por período
- Se usa para: proyectar la serie Real cuando hay meses sin IPC en el Excel, y como referencia en pronósticos

## Modelo de Scoring (0–100)

- **MAPE** (peso 40%): <2% = 95pts, <5% = 82, <10% = 65, <20% = 45, <35% = 25, >35% = 8
- **Ljung-Box** (peso 35%): p>0.20 = 95pts, >0.10 = 80, >0.05 = 65, >0.01 = 40, else = 12
- **CV-RMSE** (peso 25%): RMSE/media — <5% = 95pts, <10% = 82, <20% = 65, <35% = 45, else = 20
- Interpretación: 80-100 Excelente, 65-79 Bueno, 50-64 Aceptable, 35-49 Débil, 0-34 Revisar

## Variables de Entorno Necesarias

- `AI_INTEGRATIONS_ANTHROPIC_BASE_URL` — provisto por Replit AI Integration
- `AI_INTEGRATIONS_ANTHROPIC_API_KEY` — provisto por Replit AI Integration

## Gotchas

- Los datos se cargan con `@st.cache_data` — si se actualiza el Excel, reiniciar el servidor o usar `st.cache_data.clear()`
- Las variaciones de los sheets "Variación Nominal" y "Variación Real" del Excel contienen fórmulas no evaluadas; se recalculan desde los datos base
- El Excel debe estar en `attached_assets/` relativo a la raíz del workspace
- Los meses en español se aplican con `eje_x_espanol(fig, fechas)` en viz — NO usar `tickformat` de Plotly
- `use_container_width` está deprecado → usar `width="stretch"` en `st.plotly_chart()` y `st.dataframe()`
- pandas 2.x: usar `.ffill().bfill()`, jamás `fillna(method="ffill")`
- El CI del pronóstico es al **80%** (alpha=0.20) — más conservador que el estándar 95%, elegido deliberadamente para bandas más acotadas
- El Boletín Oficial no tiene API JSON pública — se usa Claude + DuckDuckGo como alternativa
