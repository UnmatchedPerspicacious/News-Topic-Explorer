"""
load_data.py
------------
Loads the 20 Newsgroups dataset from scikit-learn and saves it to
data/newsgroups.csv.

Dataset: 20 Newsgroups
  - 18,846 real news articles posted to 20 Usenet newsgroups in 1996-1997
  - Topics range from politics and religion to sports and technology
  - Hosted directly in scikit-learn — no external download needed
  - Original source: http://qwone.com/~jason/20Newsgroups/

Run:
    python src/load_data.py

Outputs:
    data/newsgroups.csv   — one row per article with text and category
"""

from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

OUTPUT     = Path(__file__).parent.parent / "data" / "newsgroups.csv"
SKLEARN_DATASET = "20newsgroups"   # built into scikit-learn, no download needed


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        print(f"Data already exists at {OUTPUT} — skipping.")
        print("Delete the file and re-run to refresh.\n")
        df = pd.read_csv(OUTPUT)
    else:
        print(f"Loading '{SKLEARN_DATASET}' from scikit-learn…")
        print("  Source: sklearn.datasets.fetch_20newsgroups")
        print("  18,846 real Usenet news articles across 20 categories\n")

        # Load all categories, remove headers/footers/quotes for cleaner text
        dataset = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            random_state=42,
        )

        df = pd.DataFrame({
            "text":     dataset.data,
            "category": [dataset.target_names[t] for t in dataset.target],
            "label":    dataset.target,
        })

        # Drop empty or very short articles
        df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)

        df.to_csv(OUTPUT, index=False)
        print(f"Saved {len(df):,} articles → {OUTPUT}")

    print(f"\nDataset summary:")
    print(f"  Articles   : {len(df):,}")
    print(f"  Categories : {df['category'].nunique()}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()