import streamlit as st
from src.main import HallucinationProofRAG

@st.cache_resource(show_spinner="Initializing Rules Engine & Indexes...")
def get_rag_engine():
    """
    Shared singleton instance of the RAG engine for the Streamlit app.
    This prevents the engine from being initialized multiple times across different pages.
    """
    return HallucinationProofRAG()
