from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import joblib
import pandas as pd
import numpy as np
import os
import re
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import shap

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Liga 1 Perú — Predictor Multi-Modelo",
    description="Predice xG>=1.5, Tiros>4 y Goles>=2 por equipo",
    version="3.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_XG_PATH    = "Hiperparametros_Finales/Goles_Esperadas/modelo_xgboost_xg.pkl"
MODEL_TIROS_PATH = "Hiperparametros_Finales/Tiros_Puerta/modelo_xgboost_tiros.pkl"
MODEL_GOLES_PATH = "Hiperparametros_Finales/Goles/modelo_xgboost_liga1_goles.pkl"
DATA_PATH        = "data/bd_liga1.csv"

modelo_xg      = None
modelo_tiros   = None
modelo_goles   = None
df_historico   = None
_perf_by_round = []
_eda_data        = {}
_clustering_data = {}
_forecast_data   = {}

# Columnas raw a extraer del CSV
COLS_STATS_CSV = [
    'goles', 'Posesión de pelota', 'Goles esperados (xG)', 'Tiros totales',
    'Tiros a puerta', 'Disparos al palo', 'Tiros fuera', 'Tiros bloqueados',
    'Tiros adentro del area', 'Tiros desde fuera del area', 'Fueras de juego',
    'Pases', 'Pases precisos', 'Saques de banda', 'Pases al ultimo tercio',
    'Entradas', 'Intercepciones', 'Recuperaciones', 'Despejes', 'Corners', 'Faltas',
    'Tiros libres', 'Tarjetas amarillas', 'Tarjetas rojas', 'Atajadas', 'Saques de meta',
]

# Variables para rolling (sin Pases/Pases precisos → reemplazados por ratios)
COLS_PARA_ROLLING = [
    'goles', 'Posesión de pelota', 'Goles esperados (xG)', 'Tiros totales',
    'Tiros a puerta', 'Disparos al palo', 'Tiros fuera', 'Tiros bloqueados',
    'Tiros adentro del area', 'Tiros desde fuera del area', 'Fueras de juego',
    'Saques de banda', 'Pases al ultimo tercio',
    'Entradas', 'Intercepciones', 'Recuperaciones', 'Despejes', 'Corners', 'Faltas',
    'Tiros libres', 'Tarjetas amarillas', 'Tarjetas rojas', 'Atajadas', 'Saques de meta',
    'precision_pases', 'precision_tiros', 'conversion_xg', 'ratio_area',
]

# Feature set unificado para los 3 modelos: 28*2 + Local = 57 features
FEATURES_FINAL = (
    [f'{col}_prom_3' for col in COLS_PARA_ROLLING] +
    [f'{col}_prom_5' for col in COLS_PARA_ROLLING] +
    ['Local']
)

# Panel 1 — EDA: variables ofensivas/disciplinarias clave a nivel equipo-partido
EDA_COLS = [
    'goles', 'Posesión de pelota', 'Goles esperados (xG)', 'Tiros totales',
    'Tiros a puerta', 'Corners', 'Faltas', 'Tarjetas amarillas',
]

# Panel 1 — Clustering: perfil ofensivo/defensivo promedio por equipo
CLUSTER_COLS = [
    'goles', 'Posesión de pelota', 'Goles esperados (xG)', 'Tiros totales',
    'Tiros a puerta', 'Tiros adentro del area', 'Corners', 'Faltas',
    'Tarjetas amarillas', 'Entradas', 'Intercepciones', 'Despejes',
]

# Panel 3 — Pronóstico: series mensuales combinadas (local + visitante) por partido
FORECAST_SERIES = {
    'goles': {'label': 'Goles por Partido',              'col_local': 'goles_local',                     'col_visitante': 'goles_visitante'},
    'xg':    {'label': 'Goles Esperados (xG) por Partido','col_local': 'Goles esperados (xG)_local',      'col_visitante': 'Goles esperados (xG)_visitante'},
    'tiros': {'label': 'Tiros a Puerta por Partido',      'col_local': 'Tiros a puerta_local',            'col_visitante': 'Tiros a puerta_visitante'},
}


@app.on_event("startup")
def cargar_recursos():
    global modelo_xg, modelo_tiros, modelo_goles, df_historico

    for path, attr in [
        (MODEL_XG_PATH,    'xg'),
        (MODEL_TIROS_PATH, 'tiros'),
        (MODEL_GOLES_PATH, 'goles'),
    ]:
        if os.path.exists(path):
            try:
                m = joblib.load(path)
                if attr == 'xg':      modelo_xg    = m
                elif attr == 'tiros': modelo_tiros = m
                else:                 modelo_goles = m
                print(f"Modelo {attr} cargado: {path}")
            except Exception as e:
                print(f"ERROR cargando {path}: {e}")
        else:
            print(f"ADVERTENCIA: {path} no encontrado.")

    if os.path.exists(DATA_PATH):
        try:
            df_historico = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8-sig')
            df_historico.columns = df_historico.columns.str.strip()
            df_historico['fecha'] = pd.to_datetime(
                df_historico['fecha'], format='%d/%m/%Y', errors='coerce'
            )
            print(f"Datos históricos cargados: {len(df_historico)} partidos.")
        except Exception as e:
            print(f"ERROR cargando {DATA_PATH}: {e}")
    else:
        print(f"ADVERTENCIA: {DATA_PATH} no encontrado.")

    _precompute_performance()
    _precompute_eda()
    _precompute_clustering()
    _precompute_forecast()


def _build_team_df(team_name: str, df_fuente: pd.DataFrame) -> pd.DataFrame | None:
    """Construye el historial de un equipo con ratios calculados."""
    rows = []
    for _, m in df_fuente.iterrows():
        if m.get('equipo_local') == team_name:
            suffix = '_local'
        elif m.get('equipo_visitante') == team_name:
            suffix = '_visitante'
        else:
            continue
        row = {'fecha': m['fecha']}
        for col in COLS_STATS_CSV:
            val = m.get(f'{col}{suffix}', 0)
            row[col] = pd.to_numeric(val, errors='coerce') or 0.0
        rows.append(row)

    if not rows:
        return None

    df_t = pd.DataFrame(rows).sort_values('fecha').reset_index(drop=True)

    # Ratios derivados
    df_t['precision_pases'] = (
        df_t['Pases precisos'] / df_t['Pases'].replace(0, np.nan)
    ).fillna(0)
    df_t['precision_tiros'] = (
        df_t['Tiros a puerta'] / df_t['Tiros totales'].replace(0, np.nan)
    ).fillna(0)
    df_t['conversion_xg'] = (
        df_t['goles'] / df_t['Goles esperados (xG)'].replace(0, np.nan)
    ).fillna(0)
    df_t['ratio_area'] = (
        df_t['Tiros adentro del area'] / df_t['Tiros totales'].replace(0, np.nan)
    ).fillna(0)

    return df_t


def compute_team_stats(team_name: str, is_local: int, as_of_date: pd.Timestamp | None = None) -> dict | None:
    if df_historico is None or df_historico.empty:
        return None

    df_fuente = df_historico
    if as_of_date is not None:
        df_fuente = df_historico[df_historico['fecha'] < as_of_date]

    df_t = _build_team_df(team_name, df_fuente)
    if df_t is None:
        return None

    features = {'Local': is_local}
    for col in COLS_PARA_ROLLING:
        features[f'{col}_prom_3'] = float(df_t[col].tail(3).mean())
        features[f'{col}_prom_5'] = float(df_t[col].tail(5).mean())

    return features


def _compute_stats_df(team_name: str, is_local: int, df_sub: pd.DataFrame) -> dict | None:
    df_t = _build_team_df(team_name, df_sub)
    if df_t is None:
        return None

    features = {'Local': is_local}
    for col in COLS_PARA_ROLLING:
        features[f'{col}_prom_3'] = float(df_t[col].tail(3).mean())
        features[f'{col}_prom_5'] = float(df_t[col].tail(5).mean())

    return features


def run_model(modelo, stats: dict) -> tuple[float, int]:
    X = pd.DataFrame([stats])[FEATURES_FINAL]
    prob  = float(modelo.predict_proba(X)[0][1])
    clase = int(modelo.predict(X)[0])
    return round(prob * 100, 1), clase


_explainers: dict = {}


def _get_explainer(target: str):
    modelo = {'xg': modelo_xg, 'tiros': modelo_tiros, 'goles': modelo_goles}.get(target)
    if modelo is None:
        return None
    if target not in _explainers:
        _explainers[target] = shap.TreeExplainer(modelo)
    return _explainers[target]


def explain_prediction(target: str, stats: dict, top_n: int = 8) -> dict | None:
    """SHAP local (Panel 2): qué variables empujan esta predicción puntual arriba/abajo."""
    explainer = _get_explainer(target)
    if explainer is None:
        return None

    X = pd.DataFrame([stats])[FEATURES_FINAL]
    shap_row    = explainer.shap_values(X)[0]
    base_value  = float(explainer.expected_value)

    contribs = sorted(zip(FEATURES_FINAL, shap_row), key=lambda t: abs(t[1]), reverse=True)[:top_n]

    return {
        'base_value': round(base_value, 4),
        'variables': [
            {'variable': name, 'shap': round(float(val), 5)}
            for name, val in contribs
        ],
    }


def _load_date_round_map() -> dict:
    path = "data/partidos_liga1_2026.csv"
    if not os.path.exists(path):
        return {}
    try:
        df_p = pd.read_csv(path, sep=';', encoding='utf-8-sig')
        df_p.columns = df_p.columns.str.strip()
        mapping = {}
        for _, row in df_p.iterrows():
            jornada = str(row.get('Jornada', '')).strip()
            m = re.search(r'(\d+)', jornada)
            if not m:
                continue
            rnd = int(m.group(1))
            fecha = pd.to_datetime(
                str(row.get('fecha', '')).strip(), format='%d/%m/%Y', errors='coerce'
            )
            if pd.notna(fecha):
                mapping[fecha.normalize()] = rnd
        print(f"Round map cargado: {len(set(mapping.values()))} jornadas.")
        return mapping
    except Exception as e:
        print(f"Error cargando round map: {e}")
        return {}


def _precompute_performance():
    global _perf_by_round
    if df_historico is None or any(m is None for m in [modelo_xg, modelo_tiros, modelo_goles]):
        return

    CUTOFF = pd.Timestamp('2026-04-27')
    post = df_historico[df_historico['fecha'] > CUTOFF].copy().sort_values('fecha')
    if post.empty:
        print("Rendimiento: sin datos post-corte disponibles.")
        return

    date_round = _load_date_round_map()

    records = []
    for _, match in post.iterrows():
        prior = df_historico[df_historico['fecha'] < match['fecha']]
        fecha_norm = match['fecha'].normalize()
        rnd_num = date_round.get(fecha_norm)

        for team, is_local, sfx in [
            (match.get('equipo_local'),     1, '_local'),
            (match.get('equipo_visitante'), 0, '_visitante'),
        ]:
            if not team:
                continue
            stats = _compute_stats_df(team, is_local, prior)
            if stats is None:
                continue
            try:
                _, xg_c  = run_model(modelo_xg,    stats)
                _, tir_c = run_model(modelo_tiros,  stats)
                _, gol_c = run_model(modelo_goles,  stats)
            except Exception as e:
                print(f"Rendimiento error {team}: {e}")
                continue

            def nv(col, s=sfx, r=match):
                return pd.to_numeric(r.get(f'{col}{s}', 0), errors='coerce') or 0.0

            real_xg    = float(nv('Goles esperados (xG)'))
            real_tiros = int(nv('Tiros a puerta'))
            real_goles = int(nv('goles'))

            records.append({
                'fecha':    match['fecha'],
                'rnd':      rnd_num,
                'xg_ok':   (xg_c  == 1) == (real_xg    >= 1.5),
                'tiros_ok':(tir_c == 1) == (real_tiros  >  4),
                'goles_ok':(gol_c == 1) == (real_goles  >= 2),
            })

    if not records:
        print("Rendimiento: no se pudieron generar predicciones.")
        return

    df_r = pd.DataFrame(records).sort_values('fecha')

    if date_round and df_r['rnd'].notna().any():
        df_r = df_r[df_r['rnd'].notna()].copy()
        df_r['rnd'] = df_r['rnd'].astype(int)
    else:
        dates = sorted(df_r['fecha'].unique())
        rmap, rnd, grp = {}, 13, [dates[0]]
        for i in range(1, len(dates)):
            if (dates[i] - dates[i - 1]).days <= 4:
                grp.append(dates[i])
            else:
                for d in grp: rmap[d] = rnd
                rnd += 1; grp = [dates[i]]
        for d in grp: rmap[d] = rnd
        df_r['rnd'] = df_r['fecha'].map(rmap)

    rounds_out = []
    for r, g in df_r.groupby('rnd'):
        rounds_out.append({
            'jornada':   int(r),
            'fecha':     g['fecha'].min().strftime('%d/%m/%Y'),
            'total':     len(g),
            'xg_pct':    round(float(g['xg_ok'].mean())    * 100, 1),
            'tiros_pct': round(float(g['tiros_ok'].mean()) * 100, 1),
            'goles_pct': round(float(g['goles_ok'].mean()) * 100, 1),
        })

    _perf_by_round = rounds_out
    print(f"Rendimiento precomputado: {len(rounds_out)} rondas, {len(records)} predicciones.")


def _long_team_match_df(df_fuente: pd.DataFrame) -> pd.DataFrame:
    """Reestructura el histórico a una fila por equipo por partido (local + visitante)."""
    frames = []
    for sfx in ['_local', '_visitante']:
        sub = pd.DataFrame(index=df_fuente.index)
        for col in COLS_STATS_CSV:
            colname = f'{col}{sfx}'
            sub[col] = pd.to_numeric(df_fuente[colname], errors='coerce') if colname in df_fuente.columns else np.nan
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def _precompute_eda():
    """Panel 1 — EDA: estadísticas descriptivas, histogramas, boxplots (outliers 1.5·IQR) y correlación."""
    global _eda_data
    if df_historico is None or df_historico.empty:
        return

    df_long = _long_team_match_df(df_historico)

    variables = {}
    for col in EDA_COLS:
        serie = df_long[col].dropna()
        if serie.empty:
            continue
        q1, med, q3 = serie.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = serie[(serie < low_fence) | (serie > high_fence)]
        inside = serie[(serie >= low_fence) & (serie <= high_fence)]
        whisker_low  = float(inside.min()) if not inside.empty else float(serie.min())
        whisker_high = float(inside.max()) if not inside.empty else float(serie.max())

        counts, edges = np.histogram(serie, bins=10)

        variables[col] = {
            'stats': {
                'count':  int(serie.count()),
                'mean':   round(float(serie.mean()), 2),
                'std':    round(float(serie.std()), 2),
                'min':    round(float(serie.min()), 2),
                'q1':     round(float(q1), 2),
                'median': round(float(med), 2),
                'q3':     round(float(q3), 2),
                'max':    round(float(serie.max()), 2),
            },
            'histogram': {
                'bins':   [round(float(e), 2) for e in edges],
                'counts': [int(c) for c in counts],
            },
            'boxplot': {
                'min':           round(float(serie.min()), 2),
                'q1':            round(float(q1), 2),
                'median':        round(float(med), 2),
                'q3':            round(float(q3), 2),
                'max':           round(float(serie.max()), 2),
                'whisker_low':   round(whisker_low, 2),
                'whisker_high':  round(whisker_high, 2),
                'outlier_count': int(len(outliers)),
                'outlier_pct':   round(len(outliers) / len(serie) * 100, 1),
            },
        }

    corr_df = df_long[EDA_COLS].dropna()
    corr = corr_df.corr(method='pearson').round(3)

    _eda_data = {
        'n_observaciones': int(len(df_long)),
        'variables':       variables,
        'correlacion': {
            'labels': EDA_COLS,
            'matriz': corr.values.tolist(),
        },
    }
    print(f"EDA precomputado: {len(variables)} variables, {len(df_long)} observaciones equipo-partido.")


def _valid_current_teams() -> set:
    path = "data/partidos_liga1_2026.csv"
    if not os.path.exists(path):
        return set()
    try:
        df_p = pd.read_csv(path, sep=';', encoding='utf-8-sig')
        df_p.columns = df_p.columns.str.strip()
        return (
            set(df_p['equipo_local'].str.strip().dropna()) |
            set(df_p['equipo_visitante'].str.strip().dropna())
        )
    except Exception as e:
        print(f"Error cargando equipos vigentes: {e}")
        return set()


def _team_profile_df() -> pd.DataFrame | None:
    """Perfil ofensivo/defensivo promedio por equipo vigente, usando todo el histórico disponible."""
    if df_historico is None or df_historico.empty:
        return None

    valid_teams = _valid_current_teams()

    teams: dict = {}
    for _, row in df_historico.iterrows():
        for team, sfx in [(row.get('equipo_local'), '_local'), (row.get('equipo_visitante'), '_visitante')]:
            if not team or (valid_teams and team not in valid_teams):
                continue
            if team not in teams:
                teams[team] = {c: [] for c in CLUSTER_COLS}
            for col in CLUSTER_COLS:
                v = pd.to_numeric(row.get(f'{col}{sfx}', 0), errors='coerce')
                if pd.notna(v):
                    teams[team][col].append(v)

    rows = []
    for team, cols in teams.items():
        if len(cols['goles']) < 5:  # muestra mínima para un promedio estable
            continue
        row = {'equipo': team}
        for c in CLUSTER_COLS:
            row[c] = float(np.mean(cols[c])) if cols[c] else 0.0
        rows.append(row)

    if len(rows) < 4:
        return None
    return pd.DataFrame(rows)


def _precompute_clustering():
    """Panel 1 — Clustering: K-means (k=2..8) con método del codo + coeficiente de silueta."""
    global _clustering_data
    df_teams = _team_profile_df()
    if df_teams is None:
        print("Clustering: datos insuficientes.")
        return

    X = df_teams[CLUSTER_COLS].values
    X_scaled = StandardScaler().fit_transform(X)

    n = len(df_teams)
    max_k = min(8, n - 1)
    curve = []
    best_k, best_sil = 2, -1.0
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        sil = float(silhouette_score(X_scaled, km.labels_))
        curve.append({'k': k, 'inercia': round(float(km.inertia_), 2), 'silueta': round(sil, 4)})
        if sil > best_sil:
            best_sil, best_k = sil, k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_scaled)
    df_teams = df_teams.copy()
    df_teams['cluster'] = km_final.labels_

    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    df_teams['pca_x'] = coords[:, 0]
    df_teams['pca_y'] = coords[:, 1]

    # Etiquetas legibles por nivel ofensivo relativo (goles + xG), válidas para cualquier k
    ofensive_rank = (
        df_teams.groupby('cluster')[['goles', 'Goles esperados (xG)']]
        .mean().sum(axis=1).sort_values(ascending=False)
    )
    rank_labels = [
        'Ofensivo Muy Alto', 'Ofensivo Alto', 'Ofensivo Medio-Alto', 'Ofensivo Medio',
        'Ofensivo Medio-Bajo', 'Ofensivo Bajo', 'Ofensivo Muy Bajo', 'Ofensivo Mínimo',
    ]
    cluster_label = {
        int(cl): (rank_labels[i] if i < len(rank_labels) else f'Cluster {cl}')
        for i, cl in enumerate(ofensive_rank.index)
    }

    teams_out = [
        {
            'equipo':    r['equipo'],
            'cluster':   int(r['cluster']),
            'label':     cluster_label[int(r['cluster'])],
            'x':         round(float(r['pca_x']), 3),
            'y':         round(float(r['pca_y']), 3),
            'goles_avg': round(float(r['goles']), 2),
            'xg_avg':    round(float(r['Goles esperados (xG)']), 2),
            'tiros_avg': round(float(r['Tiros a puerta']), 2),
        }
        for _, r in df_teams.iterrows()
    ]

    profiles = []
    for cl, g in df_teams.groupby('cluster'):
        profiles.append({
            'cluster':    int(cl),
            'label':      cluster_label[int(cl)],
            'size':       int(len(g)),
            'equipos':    sorted(g['equipo'].tolist()),
            'goles_avg':  round(float(g['goles'].mean()), 2),
            'xg_avg':     round(float(g['Goles esperados (xG)'].mean()), 2),
            'tiros_avg':  round(float(g['Tiros a puerta'].mean()), 2),
            'faltas_avg': round(float(g['Faltas'].mean()), 2),
        })
    profiles.sort(key=lambda p: p['goles_avg'], reverse=True)

    _clustering_data = {
        'n_equipos':     n,
        'features':      CLUSTER_COLS,
        'curva_codo':    curve,
        'best_k':        best_k,
        'silueta_best':  round(best_sil, 4),
        'teams':         teams_out,
        'perfiles':      profiles,
    }
    print(f"Clustering precomputado: {n} equipos, k óptimo={best_k} (silueta={best_sil:.3f}).")


def _build_monthly_series(col_local: str, col_visitante: str) -> pd.Series | None:
    if df_historico is None or df_historico.empty:
        return None
    total = (
        pd.to_numeric(df_historico[col_local], errors='coerce') +
        pd.to_numeric(df_historico[col_visitante], errors='coerce')
    )
    df_tmp = pd.DataFrame({'fecha': df_historico['fecha'], 'valor': total}).dropna()
    df_tmp['mes'] = df_tmp['fecha'].dt.to_period('M')
    return df_tmp.groupby('mes')['valor'].mean().sort_index()


def _fit_forecast_models(y_train: np.ndarray, n_test: int) -> dict:
    """Suavizado exponencial (Holt) + ARIMA (orden elegido por AIC en train)."""
    resultados = {
        'exponencial': {
            'nombre': 'Suavizado Exponencial (Holt)',
            'pred_test': np.asarray(ExponentialSmoothing(y_train, trend='add').fit().forecast(n_test)),
        },
    }

    best_aic, best_order = np.inf, (0, 0, 0)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    fit = ARIMA(y_train, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order = fit.aic, (p, d, q)
                except Exception:
                    continue

    resultados['arima'] = {
        'nombre': f'ARIMA{best_order}',
        'orden': list(best_order),
        'pred_test': np.asarray(ARIMA(y_train, order=best_order).fit().forecast(n_test)),
    }
    return resultados


def _precompute_forecast():
    """Panel 3 — Pronóstico: suavizado exponencial + ARIMA, MAPE/RMSE y ≥4 períodos futuros."""
    global _forecast_data
    if df_historico is None or df_historico.empty:
        return

    N_TEST, N_FORECAST = 5, 4
    resultado = {}

    for key, cfg in FORECAST_SERIES.items():
        serie = _build_monthly_series(cfg['col_local'], cfg['col_visitante'])
        if serie is None or len(serie) < N_TEST + 10:
            continue

        y = serie.values
        y_train, y_test = y[:-N_TEST], y[-N_TEST:]

        try:
            fits = _fit_forecast_models(y_train, N_TEST)
        except Exception as e:
            print(f"Pronóstico {key}: error ajustando modelos: {e}")
            continue

        modelos_out = {}
        for model_key, info in fits.items():
            pred_test = info['pred_test']
            mape = float(mean_absolute_percentage_error(y_test, pred_test) * 100)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

            # Reentrena con toda la serie disponible para el pronóstico real hacia adelante
            if model_key == 'exponencial':
                forecast_vals = np.asarray(ExponentialSmoothing(y, trend='add').fit().forecast(N_FORECAST))
            else:
                forecast_vals = np.asarray(ARIMA(y, order=tuple(info['orden'])).fit().forecast(N_FORECAST))

            band = 1.96 * rmse
            future_periods = pd.period_range(serie.index[-1] + 1, periods=N_FORECAST, freq='M')

            modelos_out[model_key] = {
                'nombre': info['nombre'],
                'mape':   round(mape, 2),
                'rmse':   round(rmse, 3),
                'forecast': [
                    {
                        'periodo': str(p),
                        'valor':   round(float(v), 2),
                        'lo':      round(float(v - band), 2),
                        'hi':      round(float(v + band), 2),
                    }
                    for p, v in zip(future_periods, forecast_vals)
                ],
            }

        mejor = min(modelos_out, key=lambda k: modelos_out[k]['mape'])

        resultado[key] = {
            'label': cfg['label'],
            'serie_historica': [
                {'periodo': str(p), 'valor': round(float(v), 2)} for p, v in serie.items()
            ],
            'n_test':      N_TEST,
            'modelos':     modelos_out,
            'mejor_modelo': mejor,
        }

    _forecast_data = resultado
    print(f"Pronóstico precomputado: {len(resultado)} series.")


@app.get("/")
@limiter.limit("60/minute")
def root(request: Request):
    return {
        "sistema": "Liga 1 Perú — Predictor Multi-Modelo",
        "version": "3.0.0",
        "modelos": {
            "xg":    "Goles esperados >= 1.5",
            "tiros": "Tiros a puerta > 4",
            "goles": "Goles >= 2",
        },
        "features": len(FEATURES_FINAL),
        "endpoints": ["/predict-match", "/match-result", "/health", "/docs"]
    }


@app.api_route("/health", methods=["GET", "HEAD"])
@limiter.limit("60/minute")
def health(request: Request):
    return {
        "modelo_xg":          modelo_xg    is not None,
        "modelo_tiros":       modelo_tiros is not None,
        "modelo_goles":       modelo_goles is not None,
        "datos_cargados":     df_historico is not None,
        "partidos_historicos": len(df_historico) if df_historico is not None else 0,
        "features_por_modelo": len(FEATURES_FINAL),
        "variables_rolling":   len(COLS_PARA_ROLLING),
        "version": "3.0.0"
    }


@app.get("/predict-match")
@limiter.limit("30/minute")
def predict_match(
    request: Request,
    home: str = Query(..., description="Nombre del equipo local"),
    away: str = Query(..., description="Nombre del equipo visitante"),
    fecha: str | None = Query(None, description="Fecha del partido DD/MM/YYYY"),
    explain: bool = Query(False, description="Incluir explicación SHAP local por variable"),
):
    if any(m is None for m in [modelo_xg, modelo_tiros, modelo_goles]):
        raise HTTPException(status_code=503, detail="Modelos no disponibles")
    if df_historico is None:
        raise HTTPException(status_code=503, detail="Datos históricos no disponibles")

    as_of = None
    if fecha:
        parsed = pd.to_datetime(fecha, format='%d/%m/%Y', errors='coerce')
        if pd.notna(parsed):
            as_of = parsed

    home_stats = compute_team_stats(home, is_local=1, as_of_date=as_of)
    away_stats = compute_team_stats(away, is_local=0, as_of_date=as_of)

    if home_stats is None:
        raise HTTPException(status_code=404, detail=f"Sin datos históricos para: {home}")
    if away_stats is None:
        raise HTTPException(status_code=404, detail=f"Sin datos históricos para: {away}")

    def predict_all(stats: dict) -> dict:
        xg_p,  xg_c  = run_model(modelo_xg,   stats)
        tir_p, tir_c = run_model(modelo_tiros, stats)
        gol_p, gol_c = run_model(modelo_goles, stats)
        result = {
            "xg":    {"probabilidad": xg_p,  "alto": xg_c  == 1},
            "tiros": {"probabilidad": tir_p, "alto": tir_c == 1},
            "goles": {"probabilidad": gol_p, "alto": gol_c == 1},
        }
        if explain:
            for target in ("xg", "tiros", "goles"):
                local_exp = explain_prediction(target, stats)
                if local_exp:
                    result[target]["shap_local"] = local_exp
        return result

    return {
        "local":     {"equipo": home, **predict_all(home_stats)},
        "visitante": {"equipo": away, **predict_all(away_stats)},
    }


@app.get("/match-result")
@limiter.limit("60/minute")
def match_result(
    request: Request,
    home: str = Query(..., description="Equipo local"),
    away: str = Query(..., description="Equipo visitante"),
):
    if df_historico is None:
        raise HTTPException(status_code=503, detail="Datos no disponibles")

    mask = (
        (df_historico['equipo_local']     == home) &
        (df_historico['equipo_visitante'] == away)
    )
    found = df_historico[mask]

    if found.empty:
        raise HTTPException(status_code=404, detail=f"Partido no encontrado: {home} vs {away}")

    row = found.sort_values('fecha', ascending=False).iloc[0]

    def team_stats(suffix: str) -> dict:
        def n(col):
            return pd.to_numeric(row.get(f'{col}{suffix}', 0), errors='coerce') or 0.0
        goles = int(n('goles'))
        xg    = round(float(n('Goles esperados (xG)')), 2)
        tiros = int(n('Tiros a puerta'))
        return {
            "goles":        goles,
            "xg":           xg,
            "tiros_puerta": tiros,
            "cumple_xg":    bool(xg >= 1.5),
            "cumple_tiros": bool(tiros > 4),
            "cumple_goles": bool(goles >= 2),
        }

    return {
        "fecha":     row['fecha'].strftime('%d/%m/%Y') if pd.notna(row['fecha']) else None,
        "local":     team_stats('_local'),
        "visitante": team_stats('_visitante'),
    }


@app.get("/modelo-info")
@limiter.limit("60/minute")
def info_modelo(request: Request):
    return {
        "modelos": {
            "xg":    {"target": "xG >= 1.5",   "features": len(FEATURES_FINAL)},
            "tiros": {"target": "Tiros >= 5",   "features": len(FEATURES_FINAL)},
            "goles": {"target": "Goles >= 2",   "features": len(FEATURES_FINAL)},
        },
        "variables_rolling": len(COLS_PARA_ROLLING),
        "ventanas":  ["prom_3 (momentum inmediato)", "prom_5 (tendencia reciente)"],
        "algoritmo": "XGBoost + SMOTE + división temporal 80/20",
        "ratios":    ["precision_pases", "precision_tiros", "conversion_xg", "ratio_area"],
    }


@app.get("/team-rankings")
@limiter.limit("60/minute")
def team_rankings(request: Request):
    if df_historico is None:
        raise HTTPException(status_code=503, detail="Datos no disponibles")

    valid_teams = _valid_current_teams()

    df_2026 = df_historico[df_historico['fecha'].dt.year == 2026]

    teams: dict = {}
    for _, row in df_2026.iterrows():
        for team, sfx in [
            (row.get('equipo_local'),     '_local'),
            (row.get('equipo_visitante'), '_visitante'),
        ]:
            if not team:
                continue
            if valid_teams and team not in valid_teams:
                continue
            if team not in teams:
                teams[team] = {'xg': [], 'tiros': [], 'goles': [], 'tiros_tot': []}

            def nv(col, s=sfx, r=row):
                return pd.to_numeric(r.get(f'{col}{s}', 0), errors='coerce') or 0.0

            xg      = nv('Goles esperados (xG)')
            tir     = nv('Tiros a puerta')
            gol     = nv('goles')
            tir_tot = nv('Tiros totales')
            if xg > 0 or tir > 0:
                teams[team]['xg'].append(xg)
                teams[team]['tiros'].append(tir)
                teams[team]['goles'].append(gol)
                teams[team]['tiros_tot'].append(tir_tot)

    result = [
        {
            'equipo':        t,
            'partidos':      len(s['xg']),
            'xg_avg':        round(float(np.mean(s['xg'])),       2),
            'tiros_avg':     round(float(np.mean(s['tiros'])),     1),
            'goles_avg':     round(float(np.mean(s['goles'])),     2),
            'tiros_tot_avg': round(float(np.mean(s['tiros_tot'])), 1),
        }
        for t, s in teams.items()
        if s['xg']
    ]
    result.sort(key=lambda x: x['xg_avg'], reverse=True)
    return result


@app.get("/model-metrics")
@limiter.limit("60/minute")
def model_metrics(request: Request):
    path = "modelos/metricas_modelos.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="metricas_modelos.json no encontrado")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo métricas: {e}")


@app.get("/shap-values")
@limiter.limit("60/minute")
def shap_values_endpoint(request: Request):
    path = "modelos/shap_values.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="shap_values.json no encontrado")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo SHAP values: {e}")


@app.get("/confusion-matrix")
@limiter.limit("60/minute")
def confusion_matrix_endpoint(request: Request):
    path = "modelos/matriz_confusion.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="matriz_confusion.json no encontrado")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo matriz de confusión: {e}")


@app.get("/forecast")
@limiter.limit("60/minute")
def forecast_endpoint(request: Request):
    if not _forecast_data:
        raise HTTPException(status_code=503, detail="Pronóstico no disponible")
    return _forecast_data


@app.get("/model-performance")
@limiter.limit("60/minute")
def model_performance(request: Request):
    if not _perf_by_round:
        return {
            "resumen": None,
            "rounds":  [],
            "message": "Sin datos post-entrenamiento. Modelos entrenados hasta 27/04/2026.",
        }
    xg_avg    = round(float(np.mean([r['xg_pct']    for r in _perf_by_round])), 1)
    tiros_avg = round(float(np.mean([r['tiros_pct'] for r in _perf_by_round])), 1)
    goles_avg = round(float(np.mean([r['goles_pct'] for r in _perf_by_round])), 1)
    return {
        "resumen": {
            "xg_accuracy":    xg_avg,
            "tiros_accuracy": tiros_avg,
            "goles_accuracy": goles_avg,
            "total_rondas":   len(_perf_by_round),
        },
        "rounds": _perf_by_round,
    }


@app.get("/eda-summary")
@limiter.limit("60/minute")
def eda_summary(request: Request):
    if not _eda_data:
        raise HTTPException(status_code=503, detail="EDA no disponible")
    return _eda_data


@app.get("/clustering")
@limiter.limit("60/minute")
def clustering_endpoint(request: Request):
    if not _clustering_data:
        raise HTTPException(status_code=503, detail="Clustering no disponible")
    return _clustering_data
