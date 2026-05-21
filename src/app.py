"""
app.py
------
Streamlit dashboard for the BERTopic News Topic Explorer.

Run:
    streamlit run src/app.py

Requires models/bertopic_model/ — run train.py first.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from bertopic import BERTopic

ROOT         = Path(__file__).parent.parent
MODEL_PATH   = ROOT / "models" / "bertopic_model"
RESULTS_PATH = ROOT / "models" / "results.json"
DATA_PATH    = ROOT / "data" / "newsgroups.csv"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Topic Explorer",
    page_icon="📰",
    layout="wide",
)

# ── Theme ──────────────────────────────────────────────────────────────────
BG     = "#0f0e17"
CARD   = "#1a1928"
ACCENT = "#e8b86d"
PURPLE = "#9b8ec4"
GREEN  = "#5fcf80"
RED    = "#e05c5c"
MUTED  = "#7a78a0"
TEXT   = "#e8e6f0"

TOPIC_COLORS = [
    "#e8b86d","#9b8ec4","#5fcf80","#e05c5c","#5bc8f5",
    "#f5a623","#bd10e0","#7ed321","#4a90e2","#ff6b6b",
    "#ffd93d","#6bcb77","#4d96ff","#ff6b9d","#c77dff",
    "#48cae4","#f4a261","#2ec4b6","#e76f51","#8338ec",
]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#2d2b45", "grid.alpha": 0.4,
})

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    .block-container {{ padding-top: 2rem; }}
    .metric-card {{
        background: {CARD}; border: 1px solid #2d2b45;
        border-radius: 12px; padding: 18px 22px; text-align: center;
    }}
    .metric-value {{ font-size: 1.8rem; font-weight: 800; color: {ACCENT}; }}
    .metric-label {{ font-size: 0.8rem; color: {MUTED}; margin-top: 4px; letter-spacing: 0.06em; }}
    .topic-card {{
        background: {CARD}; border-radius: 10px;
        padding: 16px 20px; margin-bottom: 10px; border-left: 4px solid;
    }}
    .topic-title {{ font-size: 1rem; font-weight: 700; margin-bottom: 8px; }}
    .keyword-tag {{
        display: inline-block; background: #2d2b45;
        border-radius: 6px; padding: 3px 10px; margin: 3px;
        font-size: 0.8rem; color: {TEXT};
    }}
    .result-box {{
        background: {CARD}; border: 2px solid;
        border-radius: 12px; padding: 20px; margin: 10px 0;
    }}
    h1, h2, h3 {{ color: {TEXT} !important; }}
    div[data-testid="stSidebar"] {{ background-color: {CARD}; }}
    .stTextArea textarea {{
        background-color: {CARD} !important; color: {TEXT} !important;
        border: 1px solid #2d2b45 !important; border-radius: 10px !important;
    }}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────
HUGGINGFACE_MODEL = "UnmatchedPerspicacious/News-Topic-Explorer"

@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id=HUGGINGFACE_MODEL,
        filename="bertopic_model",
        repo_type="model",
    )
    return BERTopic.load(model_path)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


model   = load_model()
results = load_results()
df      = load_data()

if model is None:
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

topics_data = results.get("topics", [])

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📰 News Topic Explorer")
st.markdown(
    "Explore topics automatically discovered from **18,000+ real Usenet news articles** "
    "using **BERTopic** — a state-of-the-art topic modelling framework combining "
    "Sentence-BERT embeddings, UMAP, and HDBSCAN."
)

# ── Metrics ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, str(results.get("n_topics", "—")), "Topics discovered"),
    (c2, f"{results.get('n_docs', 0):,}", "Articles analysed"),
    (c3, f"{results.get('n_outliers', 0):,}", "Outliers"),
    (c4, "BERTopic", "Model"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Topics", "🔍 Classify a document", "📊 Topic distribution"])

# ── Tab 1: Topic list ──────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Discovered topics")
    st.markdown(
        "Each topic is represented by its most representative keywords, "
        "extracted using c-TF-IDF — words that are frequent in the topic "
        "but rare across all other topics."
    )

    n_show = st.slider("Number of topics to show", 5, min(30, len(topics_data)), 15)

    for i, topic in enumerate(topics_data[:n_show]):
        color   = TOPIC_COLORS[i % len(TOPIC_COLORS)]
        words   = [w for w, _ in topic["words"][:8]]
        scores  = [s for _, s in topic["words"][:8]]
        keyword_html = "".join([f'<span class="keyword-tag">{w}</span>' for w in words])

        st.markdown(f"""
        <div class="topic-card" style="border-color: {color};">
            <div class="topic-title" style="color: {color};">
                Topic {topic['id']} &nbsp;·&nbsp; {topic['count']:,} documents
            </div>
            {keyword_html}
        </div>""", unsafe_allow_html=True)

    # Topic size bar chart
    st.markdown("---")
    st.markdown("#### Topic sizes")
    top_topics = sorted(topics_data, key=lambda x: x["count"], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(top_topics))]
    labels  = [f"T{t['id']}: {', '.join([w for w, _ in t['words'][:2]])}" for t in top_topics]
    ax.barh(labels[::-1], [t["count"] for t in top_topics[::-1]], color=colors[::-1], height=0.6)
    ax.set_xlabel("Number of documents")
    ax.set_title("Top 20 topics by document count", fontweight="bold", color=TEXT)
    ax.grid(axis="x")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2: Classify a document ─────────────────────────────────────────────
with tab2:
    st.markdown("#### Classify any text")
    st.markdown(
        "Paste any news article, paragraph, or sentence and BERTopic will "
        "find the closest matching topic from the ones it discovered."
    )

    example_texts = {
        "Technology": "NASA announced plans to send astronauts back to the moon using the new Artemis rocket system developed with SpaceX.",
        "Politics": "The Senate voted on a new bill addressing healthcare reform, with members deeply divided along party lines.",
        "Sports": "The team scored in the final minutes to clinch the championship in a thrilling overtime victory.",
        "Religion": "The bishop addressed the congregation about the importance of community and faith during difficult times.",
    }

    ex_col1, ex_col2 = st.columns(2)
    for i, (label, text) in enumerate(example_texts.items()):
        col = ex_col1 if i % 2 == 0 else ex_col2
        with col:
            if st.button(f"📄 {label} example"):
                st.session_state["classify_text"] = text
                st.rerun()

    input_text = st.text_area(
        "Paste text here",
        value=st.session_state.get("classify_text", ""),
        height=180,
        placeholder="Paste any news article or paragraph…",
        label_visibility="collapsed",
    )

    if st.button("Find topic →", type="primary", disabled=not input_text.strip()):
        with st.spinner("Classifying…"):
            topics_pred, probs = model.transform([input_text.strip()])
            topic_id = topics_pred[0]

        if topic_id == -1:
            st.markdown(f"""
            <div class="result-box" style="border-color: {MUTED};">
                <div style="color: {MUTED}; font-size: 1.1rem; font-weight: 700;">
                    ⚠️ Outlier — no strong topic match
                </div>
                <div style="color: {MUTED}; margin-top: 8px; font-size: 0.9rem;">
                    This text doesn't strongly match any discovered topic.
                    It may be too short, too generic, or about something
                    not well represented in the training data.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            matched = next((t for t in topics_data if t["id"] == topic_id), None)
            if matched:
                color    = TOPIC_COLORS[topic_id % len(TOPIC_COLORS)]
                words    = [w for w, _ in matched["words"][:10]]
                kw_html  = "".join([f'<span class="keyword-tag">{w}</span>' for w in words])
                st.markdown(f"""
                <div class="result-box" style="border-color: {color};">
                    <div style="color: {color}; font-size: 1.3rem; font-weight: 800;">
                        Topic {topic_id}
                    </div>
                    <div style="color: {MUTED}; font-size: 0.85rem; margin: 8px 0;">
                        {matched['count']:,} documents in this topic
                    </div>
                    <div style="margin-top: 10px;">{kw_html}</div>
                </div>""", unsafe_allow_html=True)

# ── Tab 3: Topic distribution ──────────────────────────────────────────────
with tab3:
    st.markdown("#### How topics are distributed")

    if df is not None and "category" in df.columns:
        st.markdown(
            "The 20 Newsgroups dataset has 20 known categories. "
            "This shows how many articles fall into each one."
        )
        cat_counts = df["category"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(cat_counts))]
        ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=colors[::-1], height=0.65)
        ax.set_xlabel("Number of articles")
        ax.set_title("Articles per newsgroup category", fontweight="bold", color=TEXT)
        ax.grid(axis="x")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### BERTopic vs original categories")
    st.markdown(
        f"BERTopic discovered **{results.get('n_topics', '?')} topics** from the articles "
        f"without being told the 20 categories existed. Some topics map cleanly to one "
        f"category, others blend multiple related newsgroups together."
    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='color:{MUTED}; font-size:0.8rem; text-align:center;'>"
    "Model: BERTopic · Embeddings: all-MiniLM-L6-v2 · "
    "Dataset: 20 Newsgroups (scikit-learn) · "
    "Pipeline: Sentence-BERT → UMAP → HDBSCAN → c-TF-IDF"
    "</div>",
    unsafe_allow_html=True,
)