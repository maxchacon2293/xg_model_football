# ⚽ Expected Goals (xG) Model — Football Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Data](https://img.shields.io/badge/Data-StatsBomb%20Open%20Data-purple)
![License](https://img.shields.io/badge/License-MIT-green)

> A complete machine learning pipeline for building, comparing, and evaluating Expected Goals (xG) models using real event data from StatsBomb — benchmarked against StatsBomb's own proprietary xG values.

---

## 📌 What is xG?

**Expected Goals (xG)** is the probability that a given shot results in a goal, based on historical data. It is one of the most widely used metrics in modern football analytics, adopted by clubs, broadcasters, and companies like **Stats Perform / Opta**.

An xG of **0.8** means that shot, on average, results in a goal 80% of the time. An xG of **0.05** is a low-quality chance.

---

## 🎯 Project Objectives

- Build an end-to-end xG model from raw StatsBomb event data
- Engineer meaningful spatial and contextual features from shot events
- Compare 4 ML models: Logistic Regression, Gradient Boosting, Random Forest, Neural Network
- Evaluate using **AUC-ROC** and **Brier Score** (standard in probabilistic sports models)
- Benchmark model calibration against **StatsBomb's own xG values**

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [StatsBomb Open Data](https://github.com/statsbomb/open-data) |
| Competition | La Liga 2020/21 |
| Total shots | ~839 |
| Goals | ~111 (13.2% conversion rate) |
| Access | Free via `statsbombpy` |

---

## 🔧 Features Engineered

| Feature | Description |
|---|---|
| `distance` | Euclidean distance from shot location to goal center |
| `angle` | Angle subtended by the goal from the shot position (radians) |
| `is_header` | Whether the shot was a header |
| `is_first_time` | Shot taken first-time without controlling the ball |
| `is_one_on_one` | Player was one-on-one with the goalkeeper |
| `is_open_goal` | Goalkeeper was out of position |
| `under_pressure` | Player was under pressure from a defender |
| `is_penalty` | Penalty kick |
| `is_freekick` | Direct free kick |

---

## 🤖 Models Compared

| Model | AUC-ROC | Brier Score |
|---|---|---|
| **Logistic Regression** | **0.826** | **0.085** |
| Random Forest | 0.799 | 0.090 |
| Neural Network | 0.697 | 0.104 |
| Gradient Boosting | 0.678 | 0.105 |

> Logistic Regression achieved the best AUC, consistent with the sports analytics literature where interpretable models often outperform complex ones on small datasets.

---

## 📈 Key Findings

- **Distance and angle** are the most predictive features — consistent with the spatial geometry of football
- **Open goal** and **one-on-one** situations have high predictive power despite being rare
- Our model's **calibration is comparable to StatsBomb's own xG** (see calibration curve in output)
- **Headers have significantly lower xG** than foot shots at equivalent distances

---

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/xg-model-football
cd xg-model-football

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python xg_model.py
```

**Output:**
- Console: model evaluation metrics (AUC, Brier, CV-AUC)
- `xg_analysis.png`: 6-panel analysis figure

---

## 📦 Requirements

```
statsbombpy>=1.0
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
```

---

## 📁 Project Structure

```
xg-model-football/
├── xg_model.py          # Main pipeline
├── xg_analysis.png      # Output visualization
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology Notes

- **Angle calculation** uses the arc-tangent of the goal width subtended from the shot position
- **Brier Score** is used alongside AUC because it penalizes miscalibrated probabilities — critical for any application where the actual probability matters
- **Stratified k-fold CV** (k=5) handles class imbalance (~13% goal rate)
- Calibration is compared against StatsBomb's own xG to benchmark quality

---

## 🔭 Future Work

- [ ] Add freeze frame data (defenders between shooter and goal)
- [ ] Incorporate shot technique (driven, lob, half-volley)
- [ ] Train on multi-season data for better generalization
- [ ] Build an interactive Streamlit dashboard
- [ ] Extend to post-shot xG using ball trajectory

---

## 👤 Author

**Max Ignacio Chacón Villanueva**
Data Scientist | Machine Learning Engineer
[LinkedIn](https://linkedin.com/in/max-ignacio-chacon-villanueva-287b36195)

---

## 📄 License

MIT License

> *Data provided by [StatsBomb](https://statsbomb.com/) under their open data license.*
