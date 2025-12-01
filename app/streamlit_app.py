import sys
from pathlib import Path

# Add project ROOT to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


# app/streamlit_app.py
import streamlit as st
from ingestion.loaders import *
from ingestion.processors import clean_text
from ingestion.chunker import create_chunks
from ingestion.indexer import VectorIndexer
from querying.retriever import retrieve
import json

from ingestion.indexer import VectorIndexer
import streamlit as st

# Persistent FAISS index across reruns and tab switches
if "indexer" not in st.session_state:
    st.session_state.indexer = VectorIndexer()

indexer = st.session_state.indexer


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.title("🔎 Multi-Source RAG System (End-to-End)")

tabs = st.tabs(["📥 Ingestion Tools", "❓ Querying Tools"])

# --------------------------------------------------
# TAB 1: INGESTION
# --------------------------------------------------
with tabs[0]:
    st.subheader("📥 Upload or Input Data")

    input_text = st.text_area("Direct Text Input")
    uploaded_files = st.file_uploader("Upload PDF/TXT/CSV", accept_multiple_files=True)
    url_input = st.text_input("Website URL")

    if st.button("Process & Index"):
        raw_docs = []

        if input_text.strip():
            raw_docs.append(load_direct_text(input_text))

        if uploaded_files:
            for file in uploaded_files:
                name = file.name.lower()
                if name.endswith(".pdf"):
                    raw_docs.append(load_pdf(file))
                elif name.endswith(".txt"):
                    raw_docs.append(load_txt(file))
                elif name.endswith(".csv"):
                    raw_docs.append(load_csv(file))

        if url_input.strip():
            raw_docs.append(load_url(url_input.strip()))

        all_chunks = []
        for rd in raw_docs:
            cleaned = clean_text(rd["content"])
            rd["content"] = cleaned
            chunks = create_chunks(rd)
            all_chunks.extend(chunks)

        indexer.add_documents(all_chunks)
        st.success(f"Ingested & Indexed {len(all_chunks)} chunks")

# --------------------------------------------------
# TAB 2: QUERYING
# --------------------------------------------------

with tabs[1]:
    st.subheader("❓ Ask a Question")

    query = st.text_input("Enter your query")

    if st.button("Search"):
        retrieved_docs = retrieve(query, k=5)

        st.write("### 📄 Retrieved Contexts")
        for i, d in enumerate(retrieved_docs, 1):
            st.markdown(f"**Source {i}:** *{d['source']}* (chunk {d['metadata']['chunk_index']})")
            st.write(d["chunk"])
            st.write("---")

        # Build prompt
        from querying.generator import build_rag_prompt, call_llm
        prompt = build_rag_prompt(query, retrieved_docs)

        st.write("### 🤖 LLM Prompt (Debug)")
        st.code(prompt[:2000])

        answer = call_llm(prompt)

        st.write("## 🧠 Final Answer")
        st.success(answer)
