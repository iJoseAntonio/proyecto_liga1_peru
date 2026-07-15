"""
train.py — Reentrenamiento del modelo XGBoost
Ejecutar localmente después de cada jornada:
    python train.py
Luego: git add . && git commit -m "jornada X actualizada" && git push
"""
import pandas as pd
import numpy as np
import joblib
import json
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("=" * 55)
print("REENTRENAMIENTO — Liga 1 xG Predictor")
print("=" * 55)

# ── 1. Cargar dataset y filtrar hasta 19/04/2026 (datos de entrenamiento) ──────
df = pd.read_csv('../data/bd_liga1.csv', sep=';', encoding='utf-8-sig')
print(f"Dataset total: {df.shape[0]} partidos")

FECHA_CORTE_ENTRENAMIENTO = pd.Timestamp('2026-04-27')
df['_fecha_dt'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')
df = df[df['_fecha_dt'] <= FECHA_CORTE_ENTRENAMIENTO].drop(columns=['_fecha_dt']).copy()
print(f"Dataset filtrado (hasta 27/04/2026): {df.shape[0]} partidos")

# ── 2. Reestructurar (mismo pipeline del notebook) ────────────────────────────
columnas_estadisticas = [
    'goles', 'Posesión de pelota', 'Goles esperados (xG)', 'Tiros totales',
    'Tiros a puerta', 'Disparos al palo', 'Tiros fuera', 'Tiros bloqueados',
    'Tiros adentro del area', 'Tiros desde fuera del area', 'Fueras de juego',
    'Pases', 'Pases precisos', 'Saques de banda', 'Pases al ultimo tercio',
    'Entradas', 'Intercepciones', 'Recuperaciones', 'Despejes', 'Corners',
    'Faltas', 'Tarjetas amarillas', 'Tarjetas rojas'
]

df_local = df[['fecha', 'equipo_local', 'equipo_visitante']].copy()
df_local.columns = ['Fecha', 'Equipo', 'Rival']
df_local['Local'] = 1
for col in columnas_estadisticas:
    df_local[col] = df[f'{col}_local']

df_visitante = df[['fecha', 'equipo_visitante', 'equipo_local']].copy()
df_visitante.columns = ['Fecha', 'Equipo', 'Rival']
df_visitante['Local'] = 0
for col in columnas_estadisticas:
    df_visitante[col] = df[f'{col}_visitante']

df_e = pd.concat([df_local, df_visitante], ignore_index=True)
df_e['Fecha'] = pd.to_datetime(df_e['Fecha'], format='%d/%m/%Y')
df_e = df_e.sort_values(['Equipo', 'Fecha']).reset_index(drop=True)

# ── 3. Target ─────────────────────────────────────────────────────────────────
df_e['Target'] = np.where(
    (df_e['Goles esperados (xG)'] > 1.5) &
    (df_e['Tiros a puerta'] > 4) &
    (df_e['goles'] >= 1), 1, 0
)

# ── 4. Ratios de eficiencia ───────────────────────────────────────────────────
df_e['precision_pases'] = df_e['Pases precisos'] / df_e['Pases'].replace(0, np.nan)
df_e['precision_tiros'] = df_e['Tiros a puerta'] / df_e['Tiros totales'].replace(0, np.nan)

# ── 5. Rolling windows 3+5 ────────────────────────────────────────────────────
cols_rolling = [
    'goles', 'Posesión de pelota', 'Goles esperados (xG)',
    'Tiros a puerta', 'Disparos al palo', 'Tiros fuera', 'Tiros bloqueados',
    'Tiros adentro del area', 'Tiros desde fuera del area', 'Fueras de juego',
    'Saques de banda', 'Pases al ultimo tercio',
    'Entradas', 'Intercepciones', 'Recuperaciones', 'Despejes',
    'Corners', 'Faltas', 'Tarjetas amarillas', 'Tarjetas rojas',
    'precision_pases', 'precision_tiros'
]

for col in cols_rolling:
    df_e[f'{col}_prom_3'] = df_e.groupby('Equipo')[col].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df_e[f'{col}_prom_5'] = df_e.groupby('Equipo')[col].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

df_modelo = df_e.dropna().reset_index(drop=True)

FEATURES = (
    [f'{col}_prom_3' for col in cols_rolling] +
    [f'{col}_prom_5' for col in cols_rolling] +
    ['Local']
)

X = df_modelo[FEATURES]
y = df_modelo['Target']

# ── 6. División temporal 80/20 ────────────────────────────────────────────────
fecha_corte = df_modelo['Fecha'].quantile(0.8)
train_mask = df_modelo['Fecha'] <= fecha_corte
X_train, y_train = X[train_mask], y[train_mask]
print(f"Train: {train_mask.sum()} obs | Test: {(~train_mask).sum()} obs")

# ── 7. SMOTE ──────────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Distribución tras SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# ── 8. Cargar hiperparámetros y entrenar ──────────────────────────────────────
with open('hiperparametros_optimos_78.json', 'r') as f:
    best_params = json.load(f)
best_params['verbosity'] = 0

modelo = XGBClassifier(**best_params)
modelo.fit(X_train_res, y_train_res)

# ── 9. Guardar modelo ─────────────────────────────────────────────────────────
joblib.dump(modelo, 'modelo_xgboost_liga1.pkl')
print("\n✅ modelo_xgboost_liga1.pkl guardado — listo para git push")
