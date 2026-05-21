"""
evaluate.py
-----------
Loads the trained BERTopic model and generates a 4-panel evaluation
chart saved to models/plots/evaluation.png.

Run AFTER train.py:
    python src/evaluate.py

Outputs:
    models/plots/evaluation.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from bertopic import BERTopic

ROOT         = Path(__file__).parent.parent
MODEL_PATH   = ROOT / "models" / "bertopic_model"
RESULTS_PATH = ROOT / "models" / "results.json"
DATA_PATH    = ROOT / "data" / "newsgroups.csv"
PLOTS_DIR    = ROOT / "models" / "plots"

PALETTE = {
    "bg":     "#0f0e17",
    "card":   "#1a1928",
    "accent": "#e8b86d",
    "purple": "#9b8ec4",
    "green":  "#5fcf80",
    "red":    "#e05c5c",
    "muted":  "#7a78a0",
    "text":   "#e8e6f0",
}

TOPIC_COLORS = [
    "#e8b86d","#9b8ec4","#5fcf80","#e05c5c","#5bc8f5",
    "#f5a623","#bd10e0","#7ed321","#4a90e2","#ff6b6b",
    "#ffd93d","#6bcb77","#4d96ff","#ff6b9d","#c77dff",
    "#48cae4","#f4a261","#2ec4b6","#e76f51","#8338ec",
]


def style_axes(ax, title=""):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
        spine.set_alpha(0.3)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=11,
                     fontweight="bold", pad=10)


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run `python src/train.py` first.")
        raise SystemExit(1)

    print("Loading model and results…")
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    topics_data = results["topics"]
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("News Topic Explorer — BERTopic Evaluation",
                 color=PALETTE["text"], fontsize=15, fontweight="bold", y=0.98)

    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # ── 1. Topic sizes (top 20) ────────────────────────────────────────────
    ax = axes[0]
    top20  = sorted(topics_data, key=lambda x: x["count"], reverse=True)[:20]
    labels = [f"T{t['id']}: {t['words'][0][0]}" for t in top20]
    colors = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(top20))]
    ax.barh(labels[::-1], [t["count"] for t in top20[::-1]],
            color=colors[::-1], height=0.6)
    ax.set_xlabel("Documents")
    style_axes(ax, "Top 20 Topics by Size")

    # ── 2. Top keywords per top 8 topics ──────────────────────────────────
    ax = axes[1]
    ax.axis("off")
    style_axes(ax, "Top Keywords per Topic")
    y = 0.97
    for i, topic in enumerate(topics_data[:8]):
        color = TOPIC_COLORS[i % len(TOPIC_COLORS)]
        words = ", ".join([w for w, _ in topic["words"][:5]])
        ax.text(0.02, y, f"Topic {topic['id']}:", transform=ax.transAxes,
                color=color, fontsize=9, fontweight="bold")
        ax.text(0.25, y, words, transform=ax.transAxes,
                color=PALETTE["muted"], fontsize=9)
        y -= 0.11

    # ── 3. Word score bar chart for top topic ──────────────────────────────
    ax = axes[2]
    top_topic = topics_data[0]
    words  = [w for w, _ in top_topic["words"][:10]]
    scores = [s for _, s in top_topic["words"][:10]]
    colors_bar = [PALETTE["accent"] if s == max(scores) else PALETTE["purple"] for s in scores]
    ax.barh(words[::-1], scores[::-1], color=colors_bar[::-1], height=0.6)
    ax.set_xlabel("c-TF-IDF score")
    style_axes(ax, f"Topic {top_topic['id']} — Keyword Scores")

    # ── 4. Model summary ───────────────────────────────────────────────────
    ax = axes[3]
    ax.axis("off")
    style_axes(ax, "Model Summary")
    summary = [
        ("Dataset",          "20 Newsgroups"),
        ("Articles trained", f"{results['n_docs']:,}"),
        ("Topics found",     str(results["n_topics"])),
        ("Outliers",         f"{results['n_outliers']:,}"),
        ("Embedding model",  "all-MiniLM-L6-v2"),
        ("Dim reduction",    "UMAP"),
        ("Clustering",       "HDBSCAN"),
        ("Keyword method",   "c-TF-IDF"),
    ]
    y = 0.92
    for label, val in summary:
        ax.text(0.05, y, label, transform=ax.transAxes,
                color=PALETTE["muted"], fontsize=10)
        ax.text(0.95, y, val, transform=ax.transAxes,
                color=PALETTE["accent"], fontsize=10,
                fontweight="bold", ha="right")
        y -= 0.11

    out = PLOTS_DIR / "evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"Saved evaluation plot → {out}")


if __name__ == "__main__":
    main()