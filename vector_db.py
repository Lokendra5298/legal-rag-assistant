# vector_db.py

import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: PDF Directory
pdfs_directory = "pdfs/"
os.makedirs(pdfs_directory, exist_ok=True)


# Step 2: Upload PDF
def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)

    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    return file_path


# Step 3: Load PDF
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


# Step 4: Create Text Chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)


# Step 5: Embedding Model
ollama_model_name = "nomic-embed-text"

def get_embedding_model(model_name=ollama_model_name):
    return OllamaEmbeddings(model=model_name)


# Step 6: Create FAISS Vector Database
def create_vector_db(documents):
    chunks = create_chunks(documents)
    embeddings = get_embedding_model()
    faiss_db = FAISS.from_documents(chunks, embeddings)

    return faiss_db


# Step 7: Test PDF Processing
if __name__ == "__main__":

    file_path = "human_rights.pdf"

    if os.path.exists(file_path):
        documents = load_pdf(file_path)
        print("PDF Pages:", len(documents))
        faiss_db = create_vector_db(documents)
        faiss_db.save_local("vectorstore/db_faiss")
        print("FAISS Vector Database Created Successfully!")
    else:
        print("PDF file not found:", file_path)