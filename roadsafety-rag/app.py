"""
Minimal Streamlit UI for the road-safety RAG assistant.
Run with: streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from retrieval import retrieve
from generation import generate_answer

st.set_page_config(page_title="VIC Road Safety Assistant", page_icon="🚗")

st.title("🚗 VIC Road Safety Assistant")
st.caption("Test-driven RAG assistant reproducing Walert's pipeline architecture, "
           "adapted for Australian road-safety Q&A.")

question = st.text_input("Ask a road-safety question:",
                          placeholder="Can a P1 driver use a hands-free phone?")

top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 5)
filter_stage = st.sidebar.checkbox("Filter by detected driver stage", value=True)

if question:
    with st.spinner("Retrieving relevant rules..."):
        hits = retrieve(question, top_k=top_k, filter_stage=filter_stage)

    with st.spinner("Generating answer..."):
        result = generate_answer(question, hits)

    st.subheader("Answer")
    if result["used_fallback"]:
        st.warning(result["answer"])
    else:
        st.write(result["answer"])

    with st.expander("Retrieved chunks (for debugging)"):
        for h in hits:
            st.markdown(f"**[{h['distance']:.3f}]** `{h['metadata']}`")
            st.write(h["text"])
            st.divider()
