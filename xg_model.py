# Modelo de Goles Esperados (xG) - La Liga 2020/21
# Datos: StatsBomb Open Data
# Autor: Max Chacón
#
# La idea de este proyecto es construir un modelo xG desde cero usando datos reales
# de StatsBomb. No es nada del otro mundo pero sirve para entender bien
# cómo funciona internamente lo que usan empresas como Opta.
# Comparo varios modelos y los evalúo contra el xG propio de StatsBomb.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

from statsbombpy import sb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# Configuración general
# ─────────────────────────────────────────────

# La Liga 2020/21 - tiene buena cobertura de datos en StatsBomb
COMPETITION_ID  = 11
SEASON_ID       = 90

# El centro del arco en coordenadas StatsBomb (campo de 120x80)
GOAL_CENTER_X   = 120
GOAL_CENTER_Y   = 40
GOAL_WIDTH      = 7.32  # metros, ancho reglamentario

RANDOM_STATE    = 42


# ─────────────────────────────────────────────
# 1. Carga de datos
# ─────────────────────────────────────────────

def load_shots(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Descarga todos los disparos de una competencia desde StatsBomb.
    Itera partido a partido porque la API no tiene endpoint masivo.
    """
    print(f"Cargando partidos de la competencia {competition_id}, temporada {season_id}...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"Se encontraron {len(matches)} partidos. Extrayendo disparos...")

    all_shots = []
    for match_id in matches['match_id']:
        events = sb.events(match_id=match_id)
        shots  = events[events['type'] == 'Shot'].copy()
        all_shots.append(shots)

    df = pd.concat(all_shots, ignore_index=True)
    print(f"Total disparos: {len(df)}")
    return df


# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera las variables predictoras a partir de los eventos crudos.

    Las más importantes son la distancia y el ángulo al arco.
    El resto son variables contextuales del disparo.
    """
    df = df.copy()

    # Coordenadas del disparo
    df['x'] = df['location'].apply(lambda l: l[0] if isinstance(l, list) else np.nan)
    df['y'] = df['location'].apply(lambda l: l[1] if isinstance(l, list) else np.nan)

    # Distancia euclidiana al centro del arco
    df['distance'] = np.sqrt(
        (GOAL_CENTER_X - df['x'])**2 + (GOAL_CENTER_Y - df['y'])**2
    )

    # Ángulo subtendido por el arco desde la posición del disparo
    # Uso arctan2 para calcular el ángulo que "ve" el jugador hacia el arco
    # Cuanto más grande el ángulo, más fácil el disparo en teoría
    df['angle'] = np.abs(np.arctan2(
        GOAL_WIDTH * (GOAL_CENTER_X - df['x']),
        (GOAL_CENTER_X - df['x'])**2 + (df['y'] - GOAL_CENTER_Y)**2 - (GOAL_WIDTH / 2)**2
    ))

    # Variables booleanas del contexto del disparo
    df['is_header']      = (df['shot_body_part'] == 'Head').astype(int)
    df['is_first_time']  = df['shot_first_time'].fillna(False).astype(int)
    df['is_one_on_one']  = df['shot_one_on_one'].fillna(False).astype(int)
    df['is_open_goal']   = df['shot_open_goal'].fillna(False).astype(int)
    df['under_pressure'] = df['under_pressure'].fillna(False).astype(int)
    df['is_penalty']     = (df['shot_type'] == 'Penalty').astype(int)
    df['is_freekick']    = (df['shot_type'] == 'Free Kick').astype(int)

    # Variable objetivo: 1 si es gol, 0 si no
    df['goal'] = (df['shot_outcome'] == 'Goal').astype(int)

    return df


# ─────────────────────────────────────────────
# 3. Modelos a comparar
# ─────────────────────────────────────────────

def get_models() -> dict:
    """
    Defino los modelos que voy a comparar.
    Incluyo desde regresión logística hasta una red neuronal simple
    para ver si la complejidad realmente ayuda en este problema.
    """
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=RANDOM_STATE
        ),
        'Neural Network': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE
            ))
        ])
    }


# ─────────────────────────────────────────────
# 4. Entrenamiento y evaluación
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test) -> tuple:
    """
    Entrena cada modelo y calcula las métricas principales.

    Uso AUC-ROC como métrica principal y Brier Score para evaluar
    qué tan bien calibradas están las probabilidades.
    El Brier Score es clave si queremos usar las probabilidades para algo real
    (apuestas, simulaciones, etc.) no solo para clasificar.
    """
    models  = get_models()
    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n── Evaluación de modelos ───────────────────────")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]

        auc    = roc_auc_score(y_test, y_pred)
        brier  = brier_score_loss(y_test, y_pred)

        # CV sobre el dataset completo para tener una estimación más robusta
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        cv_auc = cross_val_score(model, X_all, y_all, cv=cv, scoring='roc_auc').mean()

        results[name] = {
            'model': model, 'y_pred': y_pred,
            'auc': auc, 'brier': brier, 'cv_auc': cv_auc
        }
        print(f"  {name:<25} AUC={auc:.4f}  Brier={brier:.4f}  CV-AUC={cv_auc:.4f}")

    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\n  Mejor modelo: {best_name} (AUC={results[best_name]['auc']:.4f})")
    return results, best_name


# ─────────────────────────────────────────────
# 5. Visualizaciones
# ─────────────────────────────────────────────

def plot_analysis(results, best_name, df, X_test, y_test, save_path='xg_analysis.png'):
    """
    Genera 6 gráficos de análisis del modelo.
    El más interesante es la curva de calibración comparada con StatsBomb,
    porque muestra si nuestro modelo es tan confiable como el de ellos.
    """
    PURPLE = '#9e07ae'
    GREEN  = '#00d4aa'
    ORANGE = '#ff7b00'
    best   = results[best_name]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # Curvas ROC de todos los modelos
    ax = axes[0, 0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_pred'])
        ax.plot(fpr, tpr, label=f"{name} ({res['auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], '--', color='#484f58', linewidth=1)
    ax.set_title('Curvas ROC — Modelos xG', color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tasa de Falsos Positivos', color='#8b949e')
    ax.set_ylabel('Tasa de Verdaderos Positivos', color='#8b949e')
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')

    # Calibración: nuestro modelo vs StatsBomb
    # Esto es lo más importante — si las probabilidades están bien calibradas
    ax = axes[0, 1]
    frac, mean_p = calibration_curve(y_test, best['y_pred'], n_bins=8)
    sb_frac, sb_mean = calibration_curve(
        y_test, df.loc[X_test.index, 'shot_statsbomb_xg'], n_bins=8
    )
    ax.plot(mean_p, frac, 'o-', color=PURPLE, label=f'Nuestro {best_name}', linewidth=2)
    ax.plot(sb_mean, sb_frac, 's-', color=GREEN, label='StatsBomb xG', linewidth=2)
    ax.plot([0, 1], [0, 1], '--', color='#484f58', label='Calibración perfecta')
    ax.set_title('Calibración vs StatsBomb xG', color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('xG Predicho', color='#8b949e')
    ax.set_ylabel('Fracción de Goles Real', color='#8b949e')
    ax.legend(fontsize=9, facecolor='#21262d', labelcolor='white')

    # Importancia de variables en GBM
    # Spoiler: distancia y ángulo dominan, lo cual tiene sentido geométrico
    ax = axes[0, 2]
    gb  = results['Gradient Boosting']['model']
    features = ['distance','angle','is_header','is_first_time','is_one_on_one',
                'is_open_goal','under_pressure','is_penalty','is_freekick']
    imp = pd.Series(gb.feature_importances_, index=features).sort_values()
    colors_bar = [PURPLE if i == len(imp) - 1 else GREEN for i in range(len(imp))]
    imp.plot(kind='barh', ax=ax, color=colors_bar)
    ax.set_title('Importancia de Variables (GBM)', color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importancia', color='#8b949e')

    # Mapa de disparos coloreado por xG
    ax = axes[1, 0]
    ax.set_facecolor('#2d5a1b')
    ax.add_patch(patches.Rectangle((60, 0), 60, 80, linewidth=2, edgecolor='white', facecolor='none'))
    ax.add_patch(patches.Rectangle((60, 0), 60, 40, linewidth=1, edgecolor='white', facecolor='none', linestyle='--'))
    ax.add_patch(patches.Circle((60, 40), 10, linewidth=1, edgecolor='white', facecolor='none'))
    rng   = np.random.default_rng(RANDOM_STATE)
    signs = rng.choice([-1, 1], size=len(df))
    x_pos = 120 - df['distance'] * np.cos(df['angle'])
    y_pos = 40  + df['distance'] * np.sin(df['angle']) * signs
    sc = ax.scatter(x_pos, y_pos, c=df['shot_statsbomb_xg'],
                    cmap='RdYlGn', alpha=0.5, s=15, vmin=0, vmax=0.5)
    plt.colorbar(sc, ax=ax, label='xG')
    ax.set_xlim(60, 125)
    ax.set_ylim(0, 80)
    ax.set_title('Mapa de Disparos — Valor xG', color='white', fontsize=13, fontweight='bold')

    # Comparación de modelos en AUC y Brier
    ax = axes[1, 1]
    names_list = list(results.keys())
    aucs   = [results[n]['auc']   for n in names_list]
    briers = [results[n]['brier'] for n in names_list]
    x_idx  = np.arange(len(names_list))
    w      = 0.35
    ax.bar(x_idx - w/2, aucs,   w, label='AUC (mayor = mejor)',    color=PURPLE, alpha=0.85)
    ax.bar(x_idx + w/2, briers, w, label='Brier (menor = mejor)', color=GREEN,  alpha=0.85)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names_list], color='#c9d1d9', fontsize=8)
    ax.set_title('Comparación de Modelos', color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')

    # Distribución del xG para goles vs no-goles
    # Si el modelo es bueno, las distribuciones deberían estar bien separadas
    ax = axes[1, 2]
    ax.hist(df[df['goal'] == 0]['shot_statsbomb_xg'], bins=20, alpha=0.7,
            color='#484f58', label='No Gol', density=True)
    ax.hist(df[df['goal'] == 1]['shot_statsbomb_xg'], bins=20, alpha=0.7,
            color=ORANGE, label='Gol', density=True)
    ax.set_title('Distribución xG: Goles vs No Goles', color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('Valor xG', color='#8b949e')
    ax.set_ylabel('Densidad', color='#8b949e')
    ax.legend(fontsize=9, facecolor='#21262d', labelcolor='white')

    plt.suptitle(
        'Modelo Expected Goals (xG) — La Liga 2020/21 | StatsBomb Open Data',
        color='white', fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n  Gráfico guardado en: {save_path}")


# ─────────────────────────────────────────────
# 6. Pipeline principal
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Modelo xG — La Liga 2020/21")
    print("=" * 55)

    # Cargar y procesar datos
    raw_shots = load_shots(COMPETITION_ID, SEASON_ID)
    shots     = engineer_features(raw_shots)

    features = [
        'distance', 'angle', 'is_header', 'is_first_time',
        'is_one_on_one', 'is_open_goal', 'under_pressure',
        'is_penalty', 'is_freekick'
    ]
    df_clean = shots[features + ['goal', 'shot_statsbomb_xg']].dropna()

    print(f"\nDataset: {len(df_clean)} disparos | {df_clean['goal'].sum()} goles "
          f"({df_clean['goal'].mean()*100:.1f}% conversión)")

    X = df_clean[features]
    y = df_clean['goal']

    # 80/20 estratificado para mantener la proporción de goles
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Entrenar y evaluar
    results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Visualizar resultados
    plot_analysis(results, best_name, df_clean, X_test, y_test)

    print("\nListo.")
    print(f"  Mejor modelo : {best_name}")
    print(f"  AUC          : {results[best_name]['auc']:.4f}")
    print(f"  Brier Score  : {results[best_name]['brier']:.4f}")


if __name__ == '__main__':
    main()
