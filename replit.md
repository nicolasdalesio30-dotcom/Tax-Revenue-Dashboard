# Dashboard de Recaudación Tributaria

Dashboard interactivo en Streamlit para visualizar y analizar la recaudación tributaria argentina.

## Run & Operate

- `streamlit run artifacts/streamlit-dashboard/app.py --server.port 8000` — iniciar el dashboard
- El workflow `Dashboard Recaudación` inicia automáticamente el dashboard en el puerto 8000

## Stack

- Python 3.11 + Streamlit 1.57
- Pandas para procesamiento de datos
- Plotly para gráficos interactivos
- OpenPyXL para lectura del Excel

## Where things live

- `artifacts/streamlit-dashboard/app.py` — aplicación principal del dashboard
- `artifacts/streamlit-dashboard/.streamlit/config.toml` — configuración del servidor Streamlit
- `attached_assets/Recaudación_2023_hasta_2026_Nominal_y_Real_1778714645756.xlsx` — fuente de datos

## Estructura del Excel

- **Nominal**: recaudación en valores corrientes
- **Real**: recaudación ajustada por inflación (base 2023)
- **Variación Nominal / Real**: calculadas dinámicamente desde los datos base
- Fila 9 (Excel): fechas (columna C en adelante, mensual desde ene-2023)
- Columna B: nombres de los impuestos

## Secciones del Dashboard

1. **Visor del Último Mes**: KPIs de variación, torta de composición, tabla resumen
2. **Análisis Histórico**: serie temporal con medias móviles 3M y 6M, variaciones mensual e interanual
3. **Proyecciones**: placeholder para auto-ARIMA (próxima etapa)

## Product

Dashboard de recaudación tributaria para uso interno de la oficina. Permite:
- Ver la composición de la recaudación del último mes disponible
- Analizar la evolución histórica por impuesto (Nominal o Real)
- Visualizar variaciones mensuales e interanuales
- Seleccionar período de análisis: 6 meses, 1 año, 2 años o histórico

## Gotchas

- Los datos se cargan con `@st.cache_data` — si se actualiza el Excel, reiniciar el servidor o usar `st.cache_data.clear()`
- Las variaciones de los sheets "Variación Nominal" y "Variación Real" del Excel contienen fórmulas no evaluadas; se recalculan desde los datos base
- El Excel debe estar en `attached_assets/` relativo a la raíz del workspace
