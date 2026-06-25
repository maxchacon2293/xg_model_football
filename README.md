# ⚽ Modelo de Goles Esperados (xG) — Análisis de Fútbol

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Datos](https://img.shields.io/badge/Datos-StatsBomb%20Open%20Data-purple)
![Licencia](https://img.shields.io/badge/Licencia-MIT-green)

> Pipeline completo de machine learning para construir, comparar y evaluar modelos de Expected Goals (xG) usando datos reales de StatsBomb — comparado contra el xG propio de StatsBomb.

---

## ¿Qué es el xG?

El **Expected Goals (xG)** es la probabilidad de que un disparo termine en gol, calculada en base a datos históricos. Es una de las métricas más usadas en el análisis de fútbol moderno, adoptada por clubes, medios de comunicación y empresas como **Stats Perform / Opta**.

Un xG de **0.8** significa que ese disparo, en promedio, termina en gol el 80% de las veces. Un xG de **0.05** es una oportunidad de muy baja calidad.

---

## Objetivo del proyecto

La idea fue construir un modelo xG desde cero usando datos reales de StatsBomb para entender cómo funciona internamente lo que usan empresas como Opta. Comparo varios modelos y evalúo la calibración de las probabilidades contra el xG propio de StatsBomb.

- Construcción del pipeline completo desde los eventos crudos
- Feature engineering con variables espaciales y contextuales del disparo
- Comparación de 4 modelos: Regresión Logística, Gradient Boosting, Random Forest, Red Neuronal
- Evaluación con **AUC-ROC** y **Brier Score**
- Benchmark de calibración contra el **xG de StatsBomb**

---

## Datos

| Propiedad | Valor |
|---|---|
| Fuente | [StatsBomb Open Data](https://github.com/statsbomb/open-data) |
| Competencia | La Liga 2020/21 |
| Total disparos | ~839 |
| Goles | ~111 (13.2% de conversión) |
| Acceso | Gratis vía `statsbombpy` |

---

## Variables generadas

| Variable | Descripción |
|---|---|
| `distance` | Distancia euclidiana desde el disparo al centro del arco |
| `angle` | Ángulo subtendido por el arco desde la posición del disparo |
| `is_header` | Si el disparo fue de cabeza |
| `is_first_time` | Si fue un disparo de primera sin controlar |
| `is_one_on_one` | Si el jugador quedó mano a mano con el arquero |
| `is_open_goal` | Si el arco estaba vacío |
| `under_pressure` | Si había presión defensiva en el momento del disparo |
| `is_penalty` | Penal |
| `is_freekick` | Tiro libre directo |

---

## Resultados

| Modelo | AUC-ROC | Brier Score |
|---|---|---|
| **Regresión Logística** | **0.826** | **0.085** |
| Random Forest | 0.799 | 0.090 |
| Red Neuronal | 0.697 | 0.104 |
| Gradient Boosting | 0.678 | 0.105 |

La Regresión Logística ganó, lo cual tiene sentido: la relación entre distancia/ángulo y probabilidad de gol es bastante lineal. Los modelos más complejos no encuentran patrones adicionales suficientes para justificar su complejidad con este tamaño de dataset.

---

## Hallazgos principales

- **Distancia y ángulo** son las variables más predictivas — tiene sentido geométrico
- Las situaciones de **arco vacío** y **mano a mano** tienen alto poder predictivo aunque son poco frecuentes
- La **calibración de nuestro modelo es comparable al xG de StatsBomb** (ver curva de calibración)
- Los **remates de cabeza tienen xG significativamente menor** que los de pie a igual distancia

---

## Instalación y uso

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/xg-model-football
cd xg-model-football

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo
python xg_model.py
```

**Output:**
- Consola: métricas de evaluación (AUC, Brier, CV-AUC)
- `xg_analysis.png`: figura de análisis con 6 paneles

---

## Dependencias

```
statsbombpy>=1.0
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
```

---

## Estructura del proyecto

```
xg-model-football/
├── xg_model.py          # Pipeline principal
├── xg_analysis.png      # Visualización de resultados
├── requirements.txt
└── README.md
```

---

## Notas metodológicas

- El cálculo del ángulo usa arctan2 sobre el ancho del arco desde la posición del disparo
- El **Brier Score** complementa el AUC porque penaliza probabilidades mal calibradas — importante si se quieren usar las probabilidades para algo real
- **CV estratificado** (k=5) para manejar el desbalance de clases (~13% goles)
- La calibración se compara contra el xG de StatsBomb como referencia de calidad

---

## Ideas para seguir mejorando

- [ ] Agregar datos de freeze frame (defensores entre el jugador y el arco)
- [ ] Incorporar técnica del disparo (golpe, vaselina, volea)
- [ ] Entrenar con más temporadas para mejorar generalización
- [ ] Dashboard interactivo con Streamlit
- [ ] Extender a post-shot xG usando trayectoria del balón

---

## Autor

**Max Ignacio Chacón Villanueva**
Data Scientist | Machine Learning Engineer
[LinkedIn](https://linkedin.com/in/max-ignacio-chacon-villanueva-287b36195)

---

## Licencia

MIT

> *Datos proporcionados por [StatsBomb](https://statsbomb.com/) bajo su licencia de datos abiertos.*
