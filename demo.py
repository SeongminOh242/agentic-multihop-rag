import json
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Agentic Multi-Hop RAG Demo",
    page_icon="🔍",
    layout="wide",
)

RESULTS_PATH = Path(__file__).parent / "results" / "Oh_experiment_results.json"
DETAILS_PATH = Path(__file__).parent / "results" / "Oh_experiment_details.json"


@st.cache_data
def load_data():
    results = json.loads(RESULTS_PATH.read_text())
    details = json.loads(DETAILS_PATH.read_text())

    # Build per-sample lookup keyed by sample_id
    by_id = {}
    for trace in details["baseline"]["traces"]:
        by_id.setdefault(trace["sample_id"], {})["baseline"] = trace
    for trace in details["agentic"]["traces"]:
        by_id.setdefault(trace["sample_id"], {})["agentic"] = trace
    for trace in details["frontier_agentic"]["traces"]:
        by_id.setdefault(trace["sample_id"], {})["frontier"] = trace

    samples = [
        {
            "sample_id": sid,
            "question": v["agentic"]["question"],
            "gold_answer": v["agentic"]["gold_answer"],
            "question_type": v["agentic"]["question_type"],
            "retrieval_hop_need": v["agentic"]["retrieval_hop_need"],
            "agentic_em": v["agentic"]["exact_match"],
            "frontier_em": v["frontier"]["exact_match"],
            "baseline_em": v["baseline"]["exact_match"],
            **v,
        }
        for sid, v in by_id.items()
        if "agentic" in v and "frontier" in v and "baseline" in v
    ]

    return results, details, samples


def metric_badge(value: float) -> str:
    color = "#2ecc71" if value == 1.0 else "#e74c3c"
    label = "✓ Correct" if value == 1.0 else "✗ Wrong"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em">{label}</span>'


def render_hop_chain(hop_traces: list):
    if not hop_traces:
        st.info("No hop trace recorded (single-pass retrieval).")
        return

    for ht in hop_traces:
        hop_num = ht["hop"]
        query = ht["query"]
        docs = ht.get("top_doc_titles", [])
        decision = ht.get("decision", {})
        sufficient = ht.get("sufficient", False)

        with st.expander(f"Hop {hop_num} — *{query}*", expanded=(hop_num == 1)):
            st.markdown(f"**Query:** `{query}`")

            if docs:
                st.markdown("**Retrieved documents:**")
                for i, title in enumerate(docs, 1):
                    st.markdown(f"&nbsp;&nbsp;{i}. {title}")

            if decision:
                st.markdown("**Agent decision:**")
                col1, col2 = st.columns(2)
                col1.markdown(f"Next sub-query: `{decision.get('sub_query', '—')}`")
                status = "✅ Sufficient — generating answer" if sufficient else "🔄 Not sufficient — continue"
                col2.markdown(f"Status: {status}")


results, details, samples = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Agentic Multi-Hop RAG — Interactive Demo")
st.caption(
    "Iterative Agentic RAG for complex question answering on HotpotQA · "
    "CS 572 Final Project · Seongmin Oh, Sam Liu, Avaneesh Bhoite"
)

# ── Aggregate metrics ─────────────────────────────────────────────────────────
st.subheader("Overall Results (50 samples)")
agg_cols = st.columns(3)
systems = [
    ("Baseline RAG (Llama 3.1-8B)", results["baseline"], "#3498db"),
    ("Agentic RAG (Llama 3.1-8B)", results["agentic"], "#e67e22"),
    ("Frontier Agentic (Gemini 2.5 Flash)", results["frontier_agentic"], "#2ecc71"),
]

for col, (label, agg, color) in zip(agg_cols, systems):
    with col:
        st.markdown(
            f'<div style="border-left:4px solid {color};padding-left:10px">'
            f"<b>{label}</b></div>",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("EM", f"{agg['EM']:.0%}")
        m2.metric("F1", f"{agg['F1']:.2f}")
        m3, m4 = st.columns(2)
        m3.metric("MRR", f"{agg['MRR']:.3f}")
        m4.metric("NDCG@10", f"{agg['NDCG@10']:.3f}")
        if "avg_hops" in agg:
            st.metric("Avg hops", f"{agg['avg_hops']:.2f}")

st.divider()

# ── Question browser ──────────────────────────────────────────────────────────
st.subheader("Browse Questions")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    qtype_filter = st.selectbox(
        "Question type", ["All", "bridge", "comparison"]
    )

with filter_col2:
    hop_filter = st.selectbox(
        "Retrieval need", ["All", "needs_followup_hops", "single_hop_sufficient"]
    )

with filter_col3:
    outcome_filter = st.selectbox(
        "Frontier outcome", ["All", "Correct (EM=1)", "Wrong (EM=0)"]
    )

filtered = samples
if qtype_filter != "All":
    filtered = [s for s in filtered if s["question_type"] == qtype_filter]
if hop_filter != "All":
    filtered = [s for s in filtered if s["retrieval_hop_need"] == hop_filter]
if outcome_filter == "Correct (EM=1)":
    filtered = [s for s in filtered if s["frontier_em"] == 1.0]
elif outcome_filter == "Wrong (EM=0)":
    filtered = [s for s in filtered if s["frontier_em"] == 0.0]

st.caption(f"{len(filtered)} question(s) match filters")

if not filtered:
    st.warning("No questions match the selected filters.")
    st.stop()

question_labels = [
    f"[{s['question_type']}] {s['question'][:90]}{'…' if len(s['question']) > 90 else ''}"
    for s in filtered
]

selected_idx = st.selectbox("Select a question", range(len(filtered)), format_func=lambda i: question_labels[i])
sample = filtered[selected_idx]

st.divider()

# ── Selected question detail ───────────────────────────────────────────────────
st.subheader("Question Detail")

info_col, gold_col = st.columns([3, 1])
with info_col:
    st.markdown(f"**Question:** {sample['question']}")
    st.markdown(
        f"Type: `{sample['question_type']}` &nbsp;|&nbsp; "
        f"Retrieval need: `{sample['retrieval_hop_need']}`",
        unsafe_allow_html=True,
    )
with gold_col:
    st.success(f"Gold answer: **{sample['gold_answer']}**")

st.markdown("#### System Comparison")
sys_cols = st.columns(3)

for col, key, label, color in [
    (sys_cols[0], "baseline", "Baseline RAG (Llama 3.1-8B)", "#3498db"),
    (sys_cols[1], "agentic", "Agentic RAG (Llama 3.1-8B)", "#e67e22"),
    (sys_cols[2], "frontier", "Frontier Agentic (Gemini 2.5 Flash)", "#2ecc71"),
]:
    trace = sample[key]
    with col:
        st.markdown(
            f'<div style="border-left:4px solid {color};padding-left:8px;margin-bottom:8px">'
            f"<b>{label}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_badge(trace["exact_match"]) +
            f"&nbsp; F1: **{trace['f1']:.2f}**",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Answer:** {trace['predicted_answer']}")
        st.markdown(f"Hops: **{trace['num_hops']}** &nbsp;|&nbsp; MRR: **{trace['final_mrr']:.3f}**", unsafe_allow_html=True)

# ── Hop chain visualization ───────────────────────────────────────────────────
st.markdown("---")
tab_ag, tab_fr = st.tabs(["🔄 Agentic RAG Hop Chain (Llama)", "🚀 Frontier Hop Chain (Gemini)"])

with tab_ag:
    render_hop_chain(sample["agentic"].get("hop_traces", []))

with tab_fr:
    render_hop_chain(sample["frontier"].get("hop_traces", []))
