"""
Regenera models/metrics/matriz_confusion.json y metricas_modelos.json
comparando los 4 modelos (XGBoost, LightGBM, Random Forest, Logistic Regression)
contra TODOS los partidos post-corte de entrenamiento disponibles actualmente
(la misma ventana de validación fuera de tiempo que usa el panel de Backtesting
en main.py — así ambos paneles quedan consistentes entre sí).

Correr desde la carpeta backend/:
    python regenerate_metrics.py
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

import main as m  # reutiliza _compute_stats_df, FEATURES_FINAL, etc.

DATA_PATH = "../data/bd_liga1.csv"
BASE      = "../models/final"
OUT_DIR   = "../models/metrics"
CUTOFF    = pd.Timestamp('2026-04-27')

TARGETS = {
    'xg':    {'label': 'Goles Esperados ≥ 1.5', 'dir': 'goles_esperadas'},
    'tiros': {'label': 'Tiros a Puerta ≥ 5',    'dir': 'tiros_puerta'},
    'goles': {'label': 'Goles Anotados ≥ 2',    'dir': 'goles'},
}

MODEL_FILES = {
    'xg': {
        'XGBoost':             ('modelo_xgboost_xg.pkl', None),
        'LightGBM':            ('modelo_lightgbm_xg.pkl', None),
        'Random Forest':       ('modelo_randomforest_xg.pkl', None),
        'Logistic Regression': ('modelo_logistic_xg.pkl', 'scaler_logistic_xg.pkl'),
    },
    'tiros': {
        'XGBoost':             ('modelo_xgboost_tiros.pkl', None),
        'LightGBM':            ('modelo_lightgbm_tiros.pkl', None),
        'Random Forest':       ('modelo_randomforest_tiros.pkl', None),
        'Logistic Regression': ('modelo_logistic_tiros.pkl', 'scaler_logistic_tiros.pkl'),
    },
    'goles': {
        'XGBoost':             ('modelo_xgboost_liga1_goles.pkl', None),
        'LightGBM':            ('modelo_lightgbm_liga1_goles.pkl', None),
        'Random Forest':       ('modelo_randomforest_liga1_goles.pkl', None),
        'Logistic Regression': ('modelo_logistic_liga1_goles.pkl', 'scaler_logistic_liga1_goles.pkl'),
    },
}


def load_model(target, name):
    fname, scaler_fname = MODEL_FILES[target][name]
    d = TARGETS[target]['dir']
    model  = joblib.load(f"{BASE}/{d}/{fname}")
    scaler = joblib.load(f"{BASE}/{d}/{scaler_fname}") if scaler_fname else None
    return model, scaler


def main_run():
    df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')

    post = df[df['fecha'] > CUTOFF].copy().sort_values('fecha')
    print(f"Partidos post-corte ({CUTOFF.date()}): {len(post)}")

    rows = []
    for _, match in post.iterrows():
        prior = df[df['fecha'] < match['fecha']]
        for team, is_local, sfx in [
            (match.get('equipo_local'),     1, '_local'),
            (match.get('equipo_visitante'), 0, '_visitante'),
        ]:
            if not team:
                continue
            stats = m._compute_stats_df(team, is_local, prior)
            if stats is None:
                continue

            def nv(col, s=sfx, r=match):
                return pd.to_numeric(r.get(f'{col}{s}', 0), errors='coerce') or 0.0

            real_xg    = float(nv('Goles esperados (xG)'))
            real_tiros = int(nv('Tiros a puerta'))
            real_goles = int(nv('goles'))

            rows.append({
                'stats': stats,
                'xg':    1 if real_xg    >= 1.5 else 0,
                'tiros': 1 if real_tiros >  4   else 0,
                'goles': 1 if real_goles >= 2   else 0,
            })

    print(f"Observaciones equipo-partido evaluadas: {len(rows)}")
    X_all = pd.DataFrame([r['stats'] for r in rows])[m.FEATURES_FINAL]

    matriz_out    = {}
    metricas_out  = {}

    for target, cfg in TARGETS.items():
        y_true = np.array([r[target] for r in rows])
        positivos_pct = round(float(y_true.mean()) * 100, 1)

        modelos_conf, modelos_metric = [], []
        for name in ['XGBoost', 'LightGBM', 'Random Forest', 'Logistic Regression']:
            model, scaler = load_model(target, name)
            X = X_all.copy()
            if scaler is not None:
                X = pd.DataFrame(scaler.transform(X), columns=X.columns)

            y_pred  = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())

            accuracy  = (tp + tn) / len(y_true)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            auc       = roc_auc_score(y_true, y_proba)

            modelos_conf.append({
                'nombre': name, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'accuracy': round(accuracy, 4), 'precision': round(precision, 4),
                'recall': round(recall, 4), 'f1': round(f1, 4), 'auc_roc': round(float(auc), 4),
            })
            modelos_metric.append({
                'nombre': name, 'accuracy': round(accuracy, 4), 'auc_roc': round(float(auc), 4),
                'f1': round(f1, 4), 'precision': round(precision, 4), 'recall': round(recall, 4),
            })
            print(f"  {target:6s} {name:20s} n={len(y_true)} acc={accuracy:.4f} auc={auc:.4f}")

        matriz_out[target] = {
            'label': target, 'n_test': len(y_true), 'positivos_pct': positivos_pct, 'modelos': modelos_conf,
        }
        metricas_out[target] = {'label': cfg['label'], 'modelos': modelos_metric}

    with open(f'{OUT_DIR}/matriz_confusion.json', 'w', encoding='utf-8') as f:
        json.dump(matriz_out, f, indent=2, ensure_ascii=False)
    with open(f'{OUT_DIR}/metricas_modelos.json', 'w', encoding='utf-8') as f:
        json.dump(metricas_out, f, indent=2, ensure_ascii=False)

    print("\nArchivos regenerados: matriz_confusion.json, metricas_modelos.json")


if __name__ == '__main__':
    main_run()
