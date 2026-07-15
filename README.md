# ⚽ Liga 1 Perú — Sistema de Predicción de Estadísticas de Juego

Dashboard analítico en línea que aplica minería de datos y machine learning sobre partidos reales de la **Liga 1 de Perú** (2023–2026) para predecir el rendimiento ofensivo de los equipos, segmentarlos por perfil de juego, y proyectar tendencias futuras de la liga.

Trabajo final del curso **Minería de Datos (2026-I)** — UNMSM, Facultad de Ingeniería de Sistemas e Informática. Dr. José Alfredo Herrera Quispe.

---

## 🔗 Enlaces en vivo

| Recurso | URL |
|---|---|
| **Dashboard (frontend)** | https://brave-hill-0bd23c910.7.azurestaticapps.net |
| **API (backend)** | https://proyecto-liga1-peru.onrender.com |
| **Repositorio** | https://github.com/iJoseAntonio/proyecto_liga1_peru |

> ⚠️ El backend está hospedado en el plan gratuito de Render, que "duerme" tras ~15 min de inactividad. La primera petición después de eso puede tardar hasta 50 segundos en responder mientras el servicio despierta.

---

## 👥 Integrantes

| Nombre | Código |
|---|---|
| Jose Antonio Villanueva Ines | 22200116 |
| Miguel Angel Porras Chavez | 22200036 |
| Alexander Jesús Centeno Cerna | 22200011 |

---

## 1. El problema

Los equipos de la Liga 1 Perú generan estadísticas detalladas por partido (posesión, tiros, xG, faltas, etc.), pero esa información rara vez se traduce en un análisis accesible para hinchas, analistas o casas de apuestas locales. El proyecto responde tres preguntas concretas para cada partido futuro:

1. **¿Qué tan probable es que un equipo llegue a xG ≥ 1.5, Tiros a puerta ≥ 5, o Goles ≥ 2?** (clasificación binaria por equipo y partido)
2. **¿Qué perfiles de juego existen entre los equipos vigentes de la liga?** (clustering no supervisado)
3. **¿Hacia dónde va la tendencia de goles/xG/tiros de la liga en los próximos meses?** (series de tiempo)

---

## 2. Fuente de datos

Los datos **no provienen de Kaggle ni UCI**: son estadísticas reales de partidos de Liga 1 Perú, obtenidas mediante scraping de **Sofascore**, cubriendo desde **04/02/2023 hasta 31/05/2026** (**1,123 partidos**, 28 estadísticas por equipo y partido).

| Archivo | Contenido |
|---|---|
| `data/bd_liga1.csv` | Histórico completo de partidos con estadísticas local/visitante |
| `data/partidos_liga1_2026.csv` | Calendario de la temporada 2026 vigente (usado para identificar equipos activos) |
| `data/tabla_liga1_peru.csv` | Tabla de posiciones actual |

---

## 3. Arquitectura

```
Usuario ── navegador
   │
   ▼
frontend/  (HTML + JS vanilla + Chart.js + PapaParse)
   │  desplegado en Azure Static Web Apps
   │  fetch() a la API
   ▼
backend/main.py  (FastAPI)
   │  desplegado en Render
   │  precalcula EDA, clustering, pronóstico y rendimiento al arrancar
   │  sirve predicciones en vivo con modelos ya entrenados
   ▼
models/final/  (.pkl de los 3 modelos XGBoost en producción)
data/          (histórico de partidos)

Consultas (Panel CRUD) ── Supabase (PostgreSQL + REST API)
```

**Stack técnico:**
- **Backend:** Python 3.11, FastAPI, scikit-learn, XGBoost, SHAP, statsmodels (ARIMA/Holt), imbalanced-learn (SMOTE)
- **Frontend:** HTML/CSS/JavaScript sin frameworks, Chart.js para gráficos interactivos, PapaParse para CSV
- **CRUD:** Supabase (PostgreSQL gestionado + REST API)
- **Despliegue:** Render (API) + Azure Static Web Apps (dashboard), CI/CD vía GitHub Actions

---

## 4. Estructura del repositorio

```
proyecto_liga1_peru/
├── backend/
│   ├── main.py              # API FastAPI: predicciones, EDA, clustering, pronóstico, CRUD
│   ├── train.py              # Reentrenamiento del modelo XGBoost con corte de fecha
│   ├── requirements.txt
│   ├── runtime.txt           # Fija Python 3.11 para Render
│   └── supabase_setup.sql    # Esquema de la tabla `consultas`
├── frontend/
│   ├── index.html
│   ├── app.js                 # Lógica del dashboard (fetch a la API, renderizado de gráficos)
│   ├── styles.css
│   └── data/                  # Copia de los CSV que el frontend consume directamente
├── models/
│   ├── final/                 # Modelos .pkl REALMENTE usados en producción
│   │   ├── goles/              # XGBoost, LightGBM, Random Forest, Logistic Regression
│   │   ├── goles_esperadas/
│   │   └── tiros_puerta/
│   └── metrics/                # JSON servidos por la API (métricas, SHAP, matriz de confusión)
├── data/                       # CSV fuente (histórico, calendario 2026, tabla de posiciones)
├── notebooks/
│   ├── Entrenamiento_y_Comparacion_Modelos_Predictivos.ipynb
│   └── Seleccion_Umbrales_Target.ipynb
├── archive/legacy_models/      # Modelos/artefactos de versiones anteriores (no usados en producción)
└── .github/workflows/          # CI/CD del frontend hacia Azure Static Web Apps
```

---

## 5. El dashboard — 4 paneles

### Panel 1 — EDA & Clustering
- Estadísticas descriptivas, histogramas (bins por regla de Sturges), boxplots con detección de outliers (regla 1.5·IQR), mapa de correlación de Pearson.
- **Clustering:** K-means (k=2 a 8) sobre 12 variables ofensivas/defensivas/disciplinarias estandarizadas, con método del codo (inercia) y **coeficiente de silueta** como criterio de selección automática del k óptimo. Visualización 2D vía PCA.

### Panel 2 — Predicciones / Rendimiento
- **3 modelos de clasificación binaria** por partido: xG ≥ 1.5, Tiros a puerta ≥ 5, Goles ≥ 2.
- **4 algoritmos comparados** por target (XGBoost, LightGBM, Random Forest, Logistic Regression) — ver tabla de métricas abajo.
- **SHAP** (TreeExplainer): importancia global de variables (`summary_plot`) y explicación local por predicción concreta (`force_plot`-equivalente), respondiendo "¿por qué el modelo predijo esto para este partido?"
- **Matriz de confusión** con TP/TN/FP/FN, precisión, recall, F1 y accuracy — evaluada sobre partidos posteriores a la fecha de corte de entrenamiento (validación fuera de tiempo, no un split aleatorio).
- **SMOTE** aplicado en entrenamiento para mitigar desbalance de clases.

### Panel 3 — Pronóstico
- Series mensuales de goles/xG/tiros de toda la liga, con **ARIMA** (orden elegido por AIC) y **Suavizado Exponencial (Holt)** compitiendo por menor MAPE en backtesting.
- Pronóstico a 4 meses con bandas de confianza (±1.96·RMSE), MAPE y RMSE reportados.

### Panel 4 — Consultas (CRUD)
- CRUD completo sobre Supabase: guardar una predicción con nota, listar consultas guardadas, editar, eliminar — todo desde el navegador.

---

## 6. Modelos y resultados

**Feature set:** 57 variables — 28 estadísticas de juego, cada una como promedio móvil de los últimos 3 y 5 partidos (`_prom_3`, `_prom_5`), más la variable situacional `Local`.

**Corte de entrenamiento:** 27/04/2026 (1,078 partidos de entrenamiento, 45 partidos posteriores reservados para validación = 90 predicciones de test, contando local y visitante).

| Target | Modelo | Accuracy | F1 | AUC-ROC |
|---|---|---|---|---|
| **xG ≥ 1.5** | XGBoost | **0.656** | 0.534 | 0.654 |
| | LightGBM | 0.652 | **0.551** | 0.660 |
| | Random Forest | 0.626 | 0.538 | 0.631 |
| | Logistic Regression | 0.633 | 0.548 | **0.666** |
| **Tiros ≥ 5** | XGBoost | **0.652** | 0.542 | 0.630 |
| | LightGBM | 0.633 | **0.546** | 0.621 |
| | Random Forest | 0.611 | 0.523 | 0.625 |
| | Logistic Regression | 0.616 | 0.535 | **0.644** |
| **Goles ≥ 2** | XGBoost | 0.621 | 0.375 | 0.569 |
| | LightGBM | 0.611 | 0.384 | 0.563 |
| | Random Forest | 0.571 | 0.493 | 0.569 |
| | Logistic Regression | **0.630** | **0.536** | **0.647** |

*(En negrita, el mejor valor de cada columna por target.)*

**Modelo desplegado en producción:** XGBoost, para los 3 targets. Es un hallazgo honesto a mencionar en la presentación: **Logistic Regression obtiene el mejor AUC-ROC en los 3 targets** (y gana en las 3 métricas para "Goles ≥ 2"), pese a que XGBoost es el modelo elegido para producción — la justificación de XGBoost se basa en su mejor accuracy en 2 de 3 targets y su mayor robustez ante relaciones no lineales sin necesidad de escalar variables, no en ser categóricamente superior en todas las métricas.

---

## 7. Correr el proyecto en local

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # En Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

La API queda disponible en `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
# Edita API_URL en app.js para que apunte a http://127.0.0.1:8000
# Luego abre index.html directamente en el navegador,
# o sirve la carpeta con un servidor estático simple:
python -m http.server 5500
```

---

## 8. Limitaciones conocidas

- El clustering usa el histórico completo por equipo (2023–2026), por lo que equipos con más antigüedad en la liga tienen más partidos promediados que los recién ascendidos — un sesgo de tamaño de muestra reconocido.
- El accuracy de los modelos ronda 57–66%: refleja la dificultad inherente de predecir estadísticas de fútbol con alta varianza, no un error de implementación.
- El plan gratuito de Render introduce latencia de arranque en frío (cold start) tras periodos de inactividad.

---

## 9. Licencia y uso

Proyecto académico desarrollado con fines educativos para el curso de Minería de Datos, UNMSM-FISI 2026-I.
