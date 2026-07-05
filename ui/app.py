"""Streamlit chat UI: ask questions, see the answer plus the exact page
images the system retrieved and cited.

Run: streamlit run ui/app.py
"""

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
            pages = retriever.retrieve(question)
        with st.spinner("Reading pages and answering..."):
            answer = vlm.answer(question, pages)

        st.markdown(answer)

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
