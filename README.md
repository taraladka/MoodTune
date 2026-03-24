# 🎵 MoodTune

### AI-Powered Music Recommendation via Emotional State Mapping

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0.3-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

MoodTune skips listening history entirely. Instead, it maps **how you feel right now** — energy, happiness, rhythm, and texture — directly to Spotify audio features, then surfaces a ranked playlist from 232,000+ tracks in under 2 seconds.

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Architecture](#-architecture) · [Performance](#-performance)

---

## ✨ What Makes This Different

Traditional music recommenders learn from your past. MoodTune works from your **present** — a 5-question mood survey feeds into a machine learning pipeline that predicts target audio features, searches a clustered track database, and re-ranks results using a hybrid scoring formula weighting audio fit, popularity, and recency.

| Traditional Systems | MoodTune |
|---|---|
| Requires listening history | Works on first use |
| Locked to your past taste | Responds to your current mood |
| Black-box collaborative filter | Interpretable feature mapping |
| Cold-start problem | No cold start — mood is enough |

---

## 🚀 Quick Start

**Prerequisites:** Python 3.8+, pip, ~1 GB disk space, 4 GB RAM minimum

```bash
# 1. Clone and install
git clone https://github.com/taraladka/moodtune.git
cd moodtune
pip install -r requirements.txt

# 2. Prepare the dataset (first time only — takes ~5–10 min)
python manage.py
# → Select [3] Prepare Data

# 3. Launch
python manage.py
# → [1] Web Interface  (http://127.0.0.1:5000)
# → [2] Terminal / CLI
```

> **Note:** The `dataset/` folder must contain your Spotify CSV dataset before running data preparation. See [Datasets](#-datasets) for sources.

---

## 🎮 Usage

### Web Interface

The browser UI walks you through 5 sliders, then surfaces 12 ranked tracks with Spotify links, match scores, and playlist management tools.

| Step | Input | Maps To |
|---|---|---|
| 1 | Energy Level (0–10) | `energy`, `tempo`, `loudness` |
| 2 | Emotional Tone (0–10) | `valence` (musical positivity) |
| 3 | Rhythm (0–10) | `danceability` |
| 4 | Sound Texture (0–10) | `acousticness` |
| 5 | Genre Base | Genre filter (10 categories + Surprise Me) |

After recommendations load: ❤️ Favorite tracks · ➕ Build playlists · ▶️ Open in Spotify · 🔄 Reshuffle without re-answering · 📥 Export to CSV

### CLI Interface

Arrow-key sliders in the terminal; same recommendation engine, zero browser required.

```
Keys: ← →  adjust · Enter  confirm
r  reshuffle · n  new query · 1–12  open track in Spotify · q  quit
```

### Example Mood Profiles

```
Party Mode       Energy 10 · Happiness 10 · Dance 9 · Style 9 (Digital) · EDM
→ High-BPM dance tracks, positive valence, strong beat

Reflection Mode  Energy 2  · Happiness 1  · Dance 2 · Style 2 (Organic) · Pop
→ Melancholic ballads, acoustic, slow tempo
```

---

## 🔬 How It Works

### 1 · Mood → Audio Features

User inputs (0–10 scale) are linearly mapped to Spotify's 9-dimensional audio feature space:

```python
target_acoustic = 1.0 - (input_style / 10)
target_loudness = -35 + (input_energy / 10 * 30)   # -35 dB → -5 dB
target_tempo    = 60  + (input_energy / 10 * 120)   # 60 → 180 BPM

feature_vector = [danceability, energy, valence, tempo,
                  acousticness, instrumentalness, liveness,
                  loudness, speechiness]
```

### 2 · Cluster-Based Candidate Search

The 232k-track dataset is pre-clustered into 20 K-Means groups at data prep time. At query time, the scaled feature vector is matched to its nearest cluster — narrowing the search space before any K-NN work begins.

```
Mood input → scaled vector → nearest K-Means cluster
                                     ↓
                          K-NN search within cluster
                          (5× the requested limit for re-ranking headroom)
```

### 3 · Hybrid Scoring & Re-ranking

Each candidate track receives a composite score before final sorting:

```
final_score = (audio_match   × 0.50)
            + (popularity    × 0.30)
            + (recency_boost × 0.20)   # +20 for 2010–2025, −10 pre-2010
```

Audio features are weighted during K-NN search to emphasize perceptually important dimensions:

| Feature | Weight | Reason |
|---|---|---|
| Valence | 1.5× | Strongest mood signal |
| Energy | 1.5× | Second strongest |
| Danceability | 1.2× | Rhythmic feel |
| Tempo / Loudness | 0.8× | Supporting context |
| Instrumentalness / Liveness | 0.5× | Less mood-critical |

### Audio Feature Reference

| Feature | Range | What It Captures |
|---|---|---|
| Danceability | 0–1 | Rhythm stability, beat strength |
| Energy | 0–1 | Intensity and perceived activity |
| Valence | 0–1 | Positivity (1 = euphoric, 0 = dark) |
| Tempo | 60–200 BPM | Speed |
| Acousticness | 0–1 | Acoustic vs. electronic |
| Instrumentalness | 0–1 | Absence of vocals |
| Liveness | 0–1 | Live performance presence |
| Loudness | −60–0 dB | Perceived volume |
| Speechiness | 0–1 | Spoken-word content |

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────┐
│                 Interface Layer               │
│   Flask Web App (app.py)   CLI (recommender)  │
└─────────────────┬─────────────────────────────┘
                  │
┌─────────────────▼─────────────────────────────┐
│           MoodTuneRecommender                 │
│  predict_features()  →  recommend()           │
│  generate_report()   →  generate_oneliner()   │
└─────────────────┬─────────────────────────────┘
                  │
┌─────────────────▼─────────────────────────────┐
│               Data Layer                      │
│  tracks_clustered.parquet  (232k tracks)      │
│  kmeans.joblib             (20 clusters)      │
│  scaler.joblib             (StandardScaler)   │
└─────────────────┬─────────────────────────────┘
                  │
┌─────────────────▼─────────────────────────────┐
│         Data Preparation Pipeline             │
│  1. Dataset selection & type detection        │
│  2. Genre consistency validation              │
│  3. Audio feature imputation                  │
│  4. K-Means clustering (k=20)                 │
│  5. StandardScaler fit & save                 │
└───────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
moodtune/
├── manage.py                  # Launcher: web / CLI / data prep
├── app.py                     # Flask routes and API
├── moodtune_recommender.py    # Core engine + CLI interface
├── data_preparation.py        # Dataset processing pipeline
├── requirements.txt
│
├── data/models/               # Generated after data prep
│   ├── tracks_clustered.parquet
│   ├── kmeans.joblib
│   └── scaler.joblib
│
├── dataset/                   # Place raw CSV(s) here
└── templates/
    └── index.html             # Single-page web UI
```

---

## 📊 Performance

### Speed (232k-track dataset)

| Stage | Time |
|---|---|
| Feature prediction | ~25 ms |
| K-NN cluster search | ~312 ms |
| Hybrid scoring & sort | ~678 ms |
| **Total end-to-end** | **< 1.5 s** |

Database loads on startup in ~8–10 s (Parquet, ~45 MB). Memory footprint: ~1 GB.

### Quality

| Metric | Value |
|---|---|
| Average audio match | 85% |
| Genre match rate | 94% |
| Artist diversity | 82% (max 2 tracks per artist) |

### Why Parquet?

Storing the clustered dataset as Parquet reduces file size by ~65% vs. CSV and enables columnar reads — only audio feature columns are loaded during K-NN search.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 2.3.3 |
| ML / data | scikit-learn 1.3.0, pandas 2.0.3, NumPy 1.24.3 |
| Storage | Parquet (PyArrow 12.0.1 + fastparquet 2023.7.0) |
| Model persistence | joblib 1.3.1 |
| Frontend | Tailwind CSS 3.x, Alpine.js 3.13.3, Phosphor Icons |
| Platforms | Windows · Linux · macOS |

---

## 📦 Datasets

MoodTune works with any Spotify-style CSV containing the 9 standard audio features. Tested sources:

- [Ultimate Spotify Tracks DB — Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
- [Spotify 1.2M+ Songs — Kaggle](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)

Place downloaded CSV files in the `dataset/` directory before running data preparation.

---

## 🤝 Contributing

```bash
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

Bug reports and feature requests welcome via [GitHub Issues](https://github.com/taraladka/MoodTune/issues). Please include your Python version, OS, and a reproducible example.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.