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
DATA_PATH        = "bd_liga1.csv"

modelo_xg      = None
modelo_tiros   = None
modelo_goles   = None
df_historico   = None
_perf_by_round = []

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


def _load_date_round_map() -> dict:
    path = "partidos_liga1_2026.csv"
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
        return {
            "xg":    {"probabilidad": xg_p,  "alto": xg_c  == 1},
            "tiros": {"probabilidad": tir_p, "alto": tir_c == 1},
            "goles": {"probabilidad": gol_p, "alto": gol_c == 1},
        }

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

    valid_teams: set = set()
    partidos_path = "partidos_liga1_2026.csv"
    if os.path.exists(partidos_path):
        try:
            df_p = pd.read_csv(partidos_path, sep=';', encoding='utf-8-sig')
            df_p.columns = df_p.columns.str.strip()
            valid_teams = (
                set(df_p['equipo_local'].str.strip().dropna()) |
                set(df_p['equipo_visitante'].str.strip().dropna())
            )
        except Exception as e:
            print(f"team-rankings: error cargando partidos CSV: {e}")

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
    path = "metricas_modelos.json"
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
    path = "shap_values.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="shap_values.json no encontrado")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo SHAP values: {e}")


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
