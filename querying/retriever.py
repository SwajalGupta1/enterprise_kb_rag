# querying/retriever.py
import streamlit as st

def retrieve(query, k=5):
    indexer = st.session_state.indexer   # Use the SAME persistent indexer
    return indexer.search(query, top_k=k)
