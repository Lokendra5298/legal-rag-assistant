# frontend.py

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from vector_db import upload_pdf, load_pdf, create_chunks, get_embedding_model
from rag_pipeline import answer_query, llm_model

st.title("AI Lawyer - PDF Question Answering")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    os.makedirs("pdfs", exist_ok=True)
    upload_pdf(uploaded_file)

    file_path = os.path.join("pdfs", uploaded_file.name)

    # Step 2: Process PDF
    documents = load_pdf(file_path)
    chunks = create_chunks(documents)

    # Step 3: Create FAISS database
    embeddings = get_embedding_model("nomic-embed-text")
    faiss_db = FAISS.from_documents(chunks, embeddings)

    st.success("PDF processed successfully!")

    # Step 4: Ask Question
    query = st.text_area("Enter your question:")

    if st.button("Ask AI Lawyer"):
        if query.strip():
            docs = faiss_db.similarity_search(query, k=4)
            response = answer_query(docs, llm_model, query)

            st.chat_message("user").write(query)
            st.chat_message("assistant").write(response.content)
        else:
            st.warning("Please enter a question!")
else:
    st.info("Please upload a PDF first.")