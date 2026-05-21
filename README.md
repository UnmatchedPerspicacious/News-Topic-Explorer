# 📰 News Topic Explorer

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-20%20Newsgroups-yellow)
![Model](https://img.shields.io/badge/Model-BERTopic-orange)
![Paradigm](https://img.shields.io/badge/ML-Unsupervised%20NLP-purple)

An unsupervised NLP project that automatically discovers hidden topics in
18,000+ real news articles using **BERTopic** — combining Sentence-BERT
embeddings, UMAP, and HDBSCAN — served through an interactive Streamlit dashboard.

---

## Live demo

**[news-topic-explorer.streamlit.app](https://news-topic-explorer.streamlit.app)**

---

## What it does

- Automatically discovers topics from a corpus of news articles with no labels
- Displays each topic as a set of representative keywords
- Lets you paste any text and find its closest matching topic
- Shows topic size distribution across the dataset

---

## The dataset

**Source:** `sklearn.datasets.fetch_20newsgroups` — built into scikit-learn, no download needed.

18,846 real Usenet news articles posted to 20 newsgroups in 1996-1997, covering
politics, religion, sports, science, technology, and more.

---

## The model

**BERTopic** is a state-of-the-art topic modelling framework:

| Step | Component | Purpose |
|---|---|---|
| 1 | `all-MiniLM-L6-v2` | Embed each document into a semantic vector |
| 2 | UMAP | Reduce embedding dimensions |
| 3 | HDBSCAN | Cluster similar documents |
| 4 | c-TF-IDF | Extract representative keywords per cluster |

Unlike LDA, BERTopic understands semantic meaning — "car" and "automobile"
are treated as similar, not as different words.

---

## Run order

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/news-topic-explorer.git
cd news-topic-explorer

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Load the dataset

```bash
python src/load_data.py
```

Loads 18,846 real news articles from scikit-learn and saves to `data/newsgroups.csv`.

### 3. Train the model

```bash
python src/train.py
```

Trains BERTopic on 5,000 articles. Expected time: 5-10 minutes on CPU.
Saves the model to `models/bertopic_model/` and results to `models/results.json`.

### 4. Launch the dashboard

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`.

---

## Project structure

```
news-topic-explorer/
├── data/
│   ├── .gitkeep
│   └── newsgroups.csv         ← loaded by load_data.py (not committed)
├── models/
│   ├── .gitkeep
│   ├── bertopic_model/        ← trained model (commit this)
│   └── results.json           ← topic data (commit this)
├── src/
│   ├── load_data.py           ← loads 20 Newsgroups dataset
│   ├── train.py               ← trains BERTopic
│   └── app.py                 ← Streamlit dashboard
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).