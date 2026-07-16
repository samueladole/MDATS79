"""
RAGScope Dashboard — Page 8: Knowledge Base
===========================================
Provides an interactive interface for exploring and managing the knowledge base
used by the RAG pipeline.
"""

from __future__ import annotations

from pipeline.vectorstore import get_vector_store
import streamlit as st

st.set_page_config(page_title="Knowledge Base · RAGScope", layout="wide", page_icon="💾")
st.title("💾 Knowledge Base")
st.caption("Explore and manage the knowledge base used by the RAG pipeline.")


st.subheader("Knowledge Base Overview")
st.markdown(
    """
    The knowledge base consists of a collection of documents that the RAG pipeline
    retrieves from to answer user queries. You can explore the contents, add new
    documents, or remove existing ones.
    """
)

col1, col2 = st.columns(2)


with col1:
    with st.form("document_form"):
        st.write("Add a new document to the knowledge base:")
        st.text_input("Title", placeholder="Enter the title of the document...")
        st.text_input("Source (Optional)", placeholder="Enter the author's name...")
        st.text_area(
            "Content",
            height=150,
            placeholder="Enter the content of the document here or upload a file...",
        )
        st.file_uploader(
            "Upload", accept_multiple_files=True, type="csv"
        )
        st.form_submit_button('💾 Embed')

with col2:
    st.write("Existing documents in the knowledge base:")

    vector_store = get_vector_store()
    documents = vector_store.list_documents()
    if documents:
        for doc in documents:
            with st.container():
                st.markdown(f"**Title:** {doc['title']}")
                st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                st.markdown(f"**Content Preview:** {doc['content'][:100]}...")
                st.button("🗑️ Remove", key=f"remove_{doc['id']}")
    else:
        st.info("No documents found. Add a new document to populate the knowledge base.")
