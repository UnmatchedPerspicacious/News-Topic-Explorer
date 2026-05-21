"""
train.py
--------
Trains a BERTopic model on the 20 Newsgroups dataset.

BERTopic pipeline:
  1. Sentence-BERT embeds each document into a dense vector
  2. UMAP reduces embeddings to lower dimensions
  3. HDBSCAN clusters the reduced embeddings
  4. c-TF-IDF extracts the most representative keywords per cluster (topic)

Run AFTER load_data.py:
    python src/train.py

Expected time: 5-10 minutes on CPU.

Outputs:
    models/bertopic_model/   — saved BERTopic model
    models/results.json      — topic info, document counts, top words
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

DATA_PATH    = Path(__file__).parent.parent / "data" / "newsgroups.csv"
MODELS_DIR   = Path(__file__).parent.parent / "models"
MODEL_PATH   = MODELS_DIR / "bertopic_model"
RESULTS_PATH = MODELS_DIR / "results.json"

SAMPLE_SIZE  = 5_000   # sample for CPU-friendly training
RANDOM_SEED  = 42


def main():
    if not DATA_PATH.exists():
        print("Data not found. Run `python src/load_data.py` first.")
        raise SystemExit(1)

    print("Loading data…")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text"])

    # Sample for speed on CPU
    df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
    docs = df["text"].tolist()
    print(f"  Training on {len(docs):,} articles")

    # ── BERTopic components ────────────────────────────────────────────────
    print("\nInitialising BERTopic components…")

    # Sentence-BERT for embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # CountVectorizer — remove stopwords for cleaner topic keywords
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nTraining BERTopic…")
    print("  Step 1/4: Embedding documents with Sentence-BERT…")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)

    # ── Results ────────────────────────────────────────────────────────────
    topic_info = topic_model.get_topic_info()
    n_topics   = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = (pd.Series(topics) == -1).sum()

    print(f"\n── Results ───────────────────────────────────────────────")
    print(f"Topics found : {n_topics}")
    print(f"Outliers     : {n_outliers:,} ({n_outliers/len(docs):.1%})")
    print(f"\nTop topics:")
    print(topic_info[topic_info["Topic"] != -1].head(10).to_string())

    # ── Save ───────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(MODEL_PATH), serialization="pickle")
    print(f"\nSaved model → {MODEL_PATH}")

    # Save results for the app
    topics_data = []
    for _, row in topic_info[topic_info["Topic"] != -1].iterrows():
        topic_id = int(row["Topic"])
        words    = topic_model.get_topic(topic_id)
        topics_data.append({
            "id":    topic_id,
            "count": int(row["Count"]),
            "name":  row["Name"],
            "words": [(w, float(s)) for w, s in words[:10]],
        })

    results = {
        "n_topics":   n_topics,
        "n_outliers": int(n_outliers),
        "n_docs":     len(docs),
        "topics":     topics_data,
        "doc_topics": topics,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results → {RESULTS_PATH}")


if __name__ == "__main__":
    main()