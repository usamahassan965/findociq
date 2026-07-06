"""Streamlit chat UI: ask questions, see the answer plus the exact page
images the system retrieved and cited.

Run: streamlit run ui/app.py
"""

import time
from pathlib import Path

import streamlit as st

from findociq.config import get_settings
from findociq.generation.vlm_client import get_vlm_client
from findociq.retrieval.hybrid_retriever import HybridRetriever

st.set_page_config(page_title="FinDocIQ", page_icon="📊", layout="wide")


@st.cache_resource
def load_pipeline():
    return HybridRetriever(), get_vlm_client()


st.title("📊 FinDocIQ")
st.caption(
    "Multimodal financial document intelligence — ColQwen2.5 visual retrieval + "
    f"hybrid search + VLM generation (provider: {get_settings().vlm_provider})"
)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask about your documents (e.g. 'What drove the revenue growth in Q3?')")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retriever, vlm = load_pipeline()

    with st.chat_message("assistant"):
        with st.spinner("Retrieving pages..."):
            t0 = time.perf_counter()
            pages = retriever.retrieve(question)
            retrieval_s = time.perf_counter() - t0

        # Stream the answer token-by-token; capture time-to-first-token as we go.
        timing = {"first_token_s": None}
        t_gen = time.perf_counter()

        def _stream():
            for chunk in vlm.answer_stream(question, pages):
                if timing["first_token_s"] is None:
                    timing["first_token_s"] = time.perf_counter() - t_gen
                yield chunk

        answer = st.write_stream(_stream)
        generation_s = time.perf_counter() - t_gen

        st.caption(
            f"⏱ retrieval {retrieval_s * 1000:.0f} ms · "
            f"first token {timing['first_token_s']:.1f} s · "
            f"full answer {generation_s:.1f} s"
            if timing["first_token_s"] is not None
            else f"⏱ retrieval {retrieval_s * 1000:.0f} ms"
        )

        with st.expander(f"📄 Retrieved pages ({len(pages)})", expanded=False):
            cols = st.columns(min(len(pages), 3) or 1)
            for i, page in enumerate(pages):
                with cols[i % len(cols)]:
                    image_path = Path(page.image_path)
                    if image_path.exists():
                        st.image(str(image_path))
                    st.caption(
                        f"**{page.doc_name}** p.{page.page_number} · "
                        f"score {page.fused_score:.4f} · via {', '.join(page.sources)}"
                    )

    st.session_state.history.append({"role": "assistant", "content": answer})
