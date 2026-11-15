"""Streamlit entry point for Judex."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import csv
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from streamlit.runtime.scriptrunner import get_script_run_ctx

from judex.catalogue import OFFICIAL_BNS_URL, SECTION_BY_NUMBER, SECTIONS
from judex.classifiers import CLASSIFIERS
from judex.engine import JudexEngine
from judex.feedback import FeedbackRepository
from judex.matcher import BNSMatcher, Match

if get_script_run_ctx() is None and __name__ == "__main__":
    print("Launching Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__] + sys.argv[1:])
    sys.exit(0)


ROOT = Path(__file__).parent
DATABASE_PATH = ROOT / "data" / "judex_feedback.db"
CURATED_TRAINING_PATH = ROOT / "data" / "curated_training.csv"


st.set_page_config(page_title="Judex | BNS Helper", page_icon="⚖️", layout="wide")

BG_PATH = ROOT / "assets" / "judex_background.png"
try:
    _bg_b64 = base64.b64encode(BG_PATH.read_bytes()).decode("ascii")
    JUDEX_BG = f"url(\'data:image/png;base64,{_bg_b64}\')"
except OSError:
    JUDEX_BG = "none"

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');
      :root { --gold:#d7b35a; --gold2:#f1d88d; --blue:#285fc7; --text:#eef3ff; --muted:#a9b9d8; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp { background:transparent !important; color:var(--text); }
      [data-testid="stAppViewContainer"] { position:relative; background:transparent !important; }
      [data-testid="stAppViewContainer"]::before {
        content:""; position:fixed; inset:0;
        background: radial-gradient(circle at 15% 12%, rgba(48,96,202,.20), transparent 30%), linear-gradient(180deg, rgba(3,7,17,.56), rgba(3,8,18,.84)), __BG__;
        background-position:center; background-size:cover; background-repeat:no-repeat; z-index:0; pointer-events:none;
      }
      [data-testid="stAppViewContainer"] > .main, [data-testid="stAppViewContainer"] .main { position:relative; z-index:1; background:transparent !important; }
      [data-testid="stHeader"] { background:rgba(3,7,15,.18) !important; }
      [data-testid="stSidebar"] { background:linear-gradient(180deg, rgba(5,10,22,.92), rgba(6,12,27,.78)); backdrop-filter:blur(14px); border-right:1px solid rgba(122,160,245,.16); }
      [data-testid="stSidebar"] * { color:#dfe9ff; }
      .block-container { max-width:1180px; padding-top:1.2rem; padding-bottom:4.5rem; }
      h1,h2,h3,p,label,span { font-family:'DM Sans',sans-serif; }
      .topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.35rem; min-height:4rem; }
      .judex-brand { display:flex; align-items:center; gap:.9rem; }
      .judex-mark { display:grid; place-items:center; width:3.2rem; height:3.2rem; border-radius:14px; background:linear-gradient(145deg,#3b76e2,#102f79); border:1px solid rgba(241,216,141,.68); box-shadow:0 8px 28px rgba(20,58,137,.35), inset 0 1px rgba(255,255,255,.14); color:#fff8de; font-size:1.6rem; }
      .judex-name { color:#fcfbf6; font-family:'Playfair Display',serif; font-size:2.35rem; font-weight:800; line-height:1; }
      .judex-tagline { color:#9fb1d7; font-family:'DM Mono',monospace; font-size:.68rem; letter-spacing:.12em; margin-top:.34rem; text-transform:uppercase; }
      .brand-rule { width:54px; height:2px; margin-top:.55rem; background:linear-gradient(90deg,var(--gold),transparent); }
      .hero { position:relative; overflow:hidden; margin-bottom:1rem; padding:1.55rem 1.65rem 1.4rem; border:1px solid rgba(215,179,90,.26); border-radius:22px; background:linear-gradient(135deg,rgba(13,26,55,.84),rgba(5,10,21,.74)); box-shadow:0 22px 65px rgba(0,0,0,.27), inset 0 1px rgba(255,255,255,.05); }
      .hero::after { content:""; position:absolute; width:240px; height:240px; right:-80px; top:-105px; border-radius:50%; background:radial-gradient(circle,rgba(211,176,85,.16),transparent 66%); }
      .eyebrow { color:#d8b85f; font-family:'DM Mono',monospace; text-transform:uppercase; font-size:.67rem; letter-spacing:.13em; margin-bottom:.48rem; }
      .hero-title { font-family:'Playfair Display',serif; font-size:2.1rem; line-height:1.08; color:#fcfbf5; margin:0; font-weight:700; }
      .hero-copy { color:#b7c5df; font-size:.94rem; max-width:760px; margin-top:.65rem; line-height:1.6; }
      .stat-row { display:flex; gap:.62rem; flex-wrap:wrap; margin-top:1rem; }
      .stat-chip { display:inline-flex; align-items:center; gap:.42rem; padding:.42rem .72rem; border:1px solid rgba(118,154,230,.22); border-radius:999px; background:rgba(16,36,81,.52); color:#dfe8fb; font-size:.74rem; }
      .stat-chip b { color:var(--gold2); font-weight:700; }
      .legal-warning { display:flex; align-items:flex-start; gap:.85rem; margin:0 0 1rem; padding:.88rem 1.04rem; border:1px solid rgba(212,168,74,.56); border-left:4px solid #d4a84a; border-radius:13px; background:linear-gradient(135deg,rgba(52,39,13,.88),rgba(10,18,37,.87)); color:#ece4c9; font-size:.84rem; line-height:1.5; box-shadow:0 10px 30px rgba(0,0,0,.2); }
      .legal-warning strong { color:#f2cb67; } .warning-icon { color:#f2cb67; font-size:1.25rem; }
      .workspace-label { color:#dce7ff; font-family:'DM Mono',monospace; font-size:.96rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.55rem; text-shadow:0 1px 10px rgba(0,0,0,.22); }
      [data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(12,21,41,.75),rgba(5,10,21,.82)); border:1px solid rgba(113,150,224,.21) !important; border-radius:17px !important; box-shadow:0 14px 42px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.025); padding:.25rem; }
      [data-testid="stTextArea"] label { color:#dbe7ff; font-weight:700; font-size:.92rem; }
      .stTextArea textarea { background:rgba(4,10,22,.78); border:1px solid #304a82; border-radius:14px; color:#f4f7ff; font-size:1rem; padding:1rem 1.05rem; line-height:1.55; box-shadow:inset 0 1px 10px rgba(0,0,0,.18); }
      .classifier-primary { color:#f4e6b0; font-size:1.06rem; font-weight:800; letter-spacing:.01em; margin:.15rem 0 .8rem; }
      .stTextArea textarea::placeholder { color:#7e93b7; }
      .stTextArea textarea:focus { border-color:#6393ee; box-shadow:0 0 0 3px rgba(72,126,255,.16),0 8px 24px rgba(0,0,0,.16); }
      .stButton > button { background:linear-gradient(110deg,#143d93,#245fca); border:1px solid rgba(154,186,247,.14); border-radius:12px; box-shadow:0 10px 25px rgba(17,51,122,.32); color:white; font-weight:700; min-height:2.8rem; transition:all .18s ease; }
      .stButton > button:hover { background:linear-gradient(110deg,#173e94,#2d6bd7); transform:translateY(-1px); border-color:rgba(241,216,141,.40); color:white; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(27,61,129,.23),rgba(9,18,36,.44)); border:1px solid rgba(114,151,232,.20); border-radius:12px; padding:.62rem .82rem; }
      [data-testid="stMetricLabel"] { color:#a7bbe6; font-size:.74rem; } [data-testid="stMetricValue"] { color:#f0d889; font-family:'DM Mono',monospace; font-size:1.25rem; }
      [data-testid="stExpander"] { border:1px solid rgba(108,143,214,.20); border-radius:14px; background:rgba(8,14,28,.48); }
      [data-testid="stAlert"] { border-radius:12px; } hr { border-color:rgba(121,149,203,.16) !important; }
      @media (max-width:640px) { .block-container{padding-top:1rem;} .hero{padding:1.2rem;} .hero-title{font-size:1.65rem;} .judex-name{font-size:1.9rem;} }
    </style>
    """.replace("__BG__", JUDEX_BG),
    unsafe_allow_html=True,
)


def get_repository() -> FeedbackRepository:
    return FeedbackRepository(DATABASE_PATH)


@st.cache_resource(show_spinner=False)
def train_engine(
    approved_section_examples: tuple[tuple[str, int], ...],
    approved_classifier_examples: tuple[tuple[str, str], ...],
    curated_section_examples: tuple[tuple[str, int], ...],
) -> JudexEngine:
    """Cache the learner and rebuild it only when reviewed training data changes."""

    return JudexEngine(
        BNSMatcher(approved_examples=approved_section_examples, curated_examples=curated_section_examples),
        approved_classifier_examples=approved_classifier_examples,
    )


@st.cache_data(show_spinner=False)
def load_curated_training() -> tuple[tuple[str, int], ...]:
    if not CURATED_TRAINING_PATH.exists():
        return ()
    rows = []
    with open(CURATED_TRAINING_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row.get("complaint", "").strip()
            label = row.get("section", "").strip()
            if text and label.isdigit():
                rows.append((text, int(label)))
    return tuple(rows)


def build_engine(repository: FeedbackRepository) -> JudexEngine:
    # Every approved feedback record changes the cache key, so the learner
    # rebuilds after admin review without retraining on every click.
    return train_engine(
        tuple(repository.approved_training_examples()),
        tuple(repository.approved_classifier_examples()),
        load_curated_training(),
    )


def render_match(match: Match, rank: int) -> None:
    heading = "Most likely match" if rank == 1 else f"Alternative {rank - 1}"
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.caption(heading)
            st.subheader(f"BNS Section {match.section.number}")
            st.write(match.section.title)
            st.caption(match.section.category)
        with right:
            st.metric("Model confidence", f"{match.confidence}%")
            st.progress(match.confidence)
        if match.matched_terms:
            st.caption("Matched terms: " + ", ".join(match.matched_terms))


def render_candidate_comparison(results: list[Match]) -> None:
    """Show alternatives without overwhelming the main BNS result."""

    if len(results) < 2:
        return
    comparison = pd.DataFrame(
        {
            "BNS section": [f"{item.section.number} — {item.section.title}" for item in results],
            "Confidence": [item.confidence for item in results],
        }
    )
    with st.expander("Compare possible sections"):
        _royal_bar_chart(comparison, "BNS section", "Confidence", "BNS candidate comparison")
        st.dataframe(comparison, hide_index=True, use_container_width=True)


def _royal_bar_chart(frame: pd.DataFrame, label_col: str, value_col: str, title: str) -> None:
    """Render a clean chart that matches the royal Judex theme.

    The main UI keeps the legal artwork as its page background; charts are
    intentionally rendered on a solid dark panel so the data remains crisp.
    """
    plot = frame.sort_values(value_col, ascending=True).copy()
    n = len(plot)
    fig, ax = plt.subplots(figsize=(10.5, max(3.6, 0.48 * n)), dpi=150)
    fig.patch.set_facecolor("#070d1c")
    ax.set_facecolor("#0b1326")

    values = plot[value_col].to_numpy(dtype=float)
    bars = ax.barh(
        plot[label_col],
        values,
        height=0.56,
        color="#2455b8",
        edgecolor="#5f86d8",
        linewidth=0.7,
        alpha=0.96,
        zorder=3,
    )
    if len(bars):
        bars[-1].set_color("#d4af55")
        bars[-1].set_edgecolor("#f0d98a")

    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#f4e8c1", pad=14)
    ax.set_xlabel("Confidence / probability (%)", fontsize=9, color="#b9c8e8", labelpad=9)
    ax.grid(axis="x", color="#91a9d8", alpha=0.14, linewidth=0.8, zorder=1)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", labelsize=9, colors="#e5ecfb", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=8, colors="#aebddd", length=0, pad=5)

    max_value = float(max(values)) if len(values) else 100.0
    right_pad = max(8.0, max_value * 0.14)
    ax.set_xlim(0, min(110, max(100, max_value + right_pad)))

    for bar, value in zip(bars, values):
        ax.text(
            min(value + max(1.5, max_value * 0.018), ax.get_xlim()[1] - 1),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#f7e8b8" if bar is bars[-1] else "#dbe6ff",
            zorder=4,
        )

    ax.axvline(0, color="#d4af55", alpha=0.65, linewidth=1.1, zorder=4)
    ax.margins(y=0.16)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.84, bottom=0.19)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def render_classifier_graph(probabilities: dict[str, float]) -> None:
    """Display the first stage before presenting BNS section results."""
    classifier_frame = pd.DataFrame(
        {
            "Classifier": list(probabilities),
            "Probability": [round(value * 100, 1) for value in probabilities.values()],
        }
    ).sort_values("Probability", ascending=False)
    top = classifier_frame.iloc[0]
    st.markdown("### Complaint classification")
    st.markdown(f"<div class='classifier-primary'>Primary classifier · {top['Classifier']} ({top['Probability']:.1f}%)</div>", unsafe_allow_html=True)
    _royal_bar_chart(classifier_frame, "Classifier", "Probability", "Complaint category confidence")
    with st.expander("View classifier scores"):
        st.dataframe(classifier_frame, hide_index=True, use_container_width=True)


def render_feedback(
    repository: FeedbackRepository,
    input_text: str,
    suggested_classifier: str,
    suggested_section: int,
) -> None:
    st.divider()
    st.subheader("Help Judex improve")
    st.caption(
        "Feedback is stored as pending. It affects the model only after an admin reviews and approves it."
    )
    classifier_options = {item.name: item.name for item in CLASSIFIERS}
    options = {f"Section {item.number} — {item.title}": item.number for item in SECTIONS}
    suggested_label = next(
        label for label, number in options.items() if number == suggested_section
    )
    with st.form("feedback_form", clear_on_submit=True):
        st.markdown("**1. Complaint classifier**")
        classifier_verdict = st.radio(
            "Was the complaint category correct?", ("Correct", "Incorrect"), horizontal=True
        )
        selected_classifier = st.selectbox(
            "Correct complaint category", list(classifier_options),
            index=list(classifier_options).index(suggested_classifier),
        )
        st.markdown("**2. BNS section match**")
        verdict = st.radio("Was the BNS section correct?", ("Correct", "Incorrect"), horizontal=True)
        selected_label = st.selectbox(
            "Correct BNS section", list(options), index=list(options).index(suggested_label)
        )
        submitted = st.form_submit_button("Submit feedback")
    if submitted:
        selected_section = options[selected_label]
        repository.submit(
            input_text=input_text,
            suggested_section=suggested_section,
            selected_section=selected_section,
            verdict=verdict.lower(),
            suggested_classifier=suggested_classifier,
            selected_classifier=selected_classifier,
            classifier_verdict=classifier_verdict.lower(),
        )
        st.success("Thanks. Your feedback is awaiting admin review.")


def render_admin(repository: FeedbackRepository) -> None:
    st.subheader("Admin feedback review")
    st.caption(
        f"{repository.count_by_status('pending')} pending · "
        f"{repository.count_by_status('approved')} approved training examples"
    )
    pending_rows = repository.pending()
    if not pending_rows:
        st.info("No feedback is waiting for review.")
        return
    for row in pending_rows:
        with st.container(border=True):
            st.caption(f"Feedback #{row['id']} · {row['created_at']}")
            st.write(row["input_text"])
            suggested = (
                f"Section {row['suggested_section']}"
                if row["suggested_section"] is not None
                else "No suggested section"
            )
            selected = SECTION_BY_NUMBER.get(row["selected_section"])
            st.caption(
                f"User verdict: {row['verdict']} · Suggested: {suggested} · "
                f"Selected: Section {selected.number} — {selected.title}"
            )
            st.caption(
                f"Classifier: {row['suggested_classifier'] or '—'} → "
                f"{row['selected_classifier'] or '—'} "
                f"({row['classifier_verdict'] or '—'})"
            )
            approve, reject = st.columns(2)
            if approve.button("Approve for training", key=f"approve_{row['id']}"):
                repository.set_status(row["id"], "approved")
                st.rerun()
            if reject.button("Reject", key=f"reject_{row['id']}"):
                repository.set_status(row["id"], "rejected")
                st.rerun()


left, right = st.columns([7, 1])
with left:
    st.markdown(
        """
        <div class="topbar">
          <div class="judex-brand">
            <div class="judex-mark">⚖</div>
            <div>
              <div class="judex-name">Judex</div>
              <div class="judex-tagline">BNS Section Discovery</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    with st.popover("About"):
        st.caption("How Judex learns")
        st.write(
            "Judex combines weighted keywords, text similarity, and a small ML "
            "classifier trained only on admin-approved feedback."
        )
        st.caption(f"Catalogue: {len(SECTIONS)} BNS sections · curated complaint training enabled.")
        st.markdown(f"[Official BNS source]({OFFICIAL_BNS_URL})")

st.markdown(
    f"""
    <section class="hero">
      <div class="eyebrow">Bharatiya Nyaya Sanhita · intelligent discovery</div>
      <div class="hero-title">Turn a complaint into a clearer legal starting point.</div>
      <div class="hero-copy">Describe the incident in plain English. Judex analyses the language, identifies a complaint category, and ranks possible BNS sections for review.</div>
      <div class="stat-row">
        <span class="stat-chip"><b>{len(SECTIONS)}</b> BNS sections</span>
        <span class="stat-chip"><b>{len(CLASSIFIERS)}</b> complaint categories</span>
        <span class="stat-chip"><b>AI-assisted</b> classification</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="legal-warning">
      <div class="warning-icon">⚠</div>
      <div>
        <strong>Important legal notice</strong><br>
        Judex is a classification and information helper only. It is not a substitute for a lawyer
        and does not provide a definitive determination of which Bharatiya Nyaya Sanhita (BNS)
        provision applies. Please consult a qualified legal professional for legal advice.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="workspace-label">Describe the incident</div>', unsafe_allow_html=True)
with st.container(border=True):
    input_text = st.text_area(
        "Describe what happened in one or more English sentences",
        placeholder="Example: Someone pretended to be my bank manager and took money from me.",
        height=140,
    )
    analyse = st.button("Analyse complaint  →", type="primary", use_container_width=True)
    if analyse:
        if len(input_text.strip()) < 12:
            st.error("Please enter a little more detail (at least 12 characters).")
        else:
            repository = get_repository()
            engine = build_engine(repository)
            analysis = engine.analyse(input_text)
            results = analysis.section_matches
            best = results[0] if results else None
            render_classifier_graph(analysis.classifier_probabilities)
            if not engine.matcher.is_confident(best, results[1] if len(results) > 1 else None):
                st.info(
                    "Judex could not confidently identify a supported BNS section from this "
                    "description. Add facts such as the action, threats/force used, property, "
                    "relationship, or injury."
                )
                if results:
                    with st.expander("Show low-confidence possibilities"):
                        for index, result in enumerate(results, start=1):
                            render_match(result, index)
            else:
                st.success("Judex found the following possible BNS match(es).")
                for index, result in enumerate(results, start=1):
                    render_match(result, index)
                render_candidate_comparison(results)
                top_classifier = max(
                    analysis.classifier_probabilities,
                    key=analysis.classifier_probabilities.get,
                )
                render_feedback(repository, input_text, top_classifier, best.section.number)

with st.sidebar:
    st.header("Judex admin")
    configured_password = os.getenv("JUDEX_ADMIN_PASSWORD")
    if not configured_password:
        st.caption("Set `JUDEX_ADMIN_PASSWORD` to enable feedback review.")
    elif st.text_input("Admin password", type="password") == configured_password:
        render_admin(get_repository())
    