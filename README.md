# AI Lawyer — PDF Question Answering with RAG

A simple **retrieval-augmented generation (RAG)** application for asking questions about the contents of an uploaded PDF. It uses **Streamlit** for the user interface, **PDFPlumber** for text extraction, **LangChain** for orchestration, **Ollama** for local text embeddings, **FAISS** for similarity search, and the **Groq API** for answer generation.

> **Disclaimer:** This is an educational document-question-answering application, not a lawyer or a substitute for professional legal advice. AI-generated responses can be incomplete or incorrect; check important statements against the original document.

## Project structure

```text
AI_Lawyer/
├── frontend.py       # Streamlit interface and PDF-to-FAISS workflow
├── vector_db.py      # PDF loading, chunking, embeddings, and FAISS helpers
├── rag_pipeline.py   # Retrieval helpers, prompt, and Groq-based generation
├── requirements.txt  # Python dependencies
├── .env               # Your local Groq API key (do not commit)
└── pdfs/              # Uploaded PDF files (created automatically)
```

## Prerequisites

- Python 3.10 or newer is recommended.
- [Ollama](https://ollama.com/) installed and running locally.
- A [Groq API](https://console.groq.com/) key with access to the configured model.
- Internet connectivity for Groq API requests.

## Installation

### 1. Clone or download the project

```bash
git clone <your-repository-url>
cd AI_Lawyer
```

If you have not published a Git repository, simply open a terminal in the folder containing `frontend.py`.

### 2. Create and activate a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

After installing Ollama, download the embedding model:

```bash
ollama pull nomic-embed-text
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Keep the server running while using the application. On some systems, the Ollama application starts the server automatically; in that case, do not start a second instance.

### 5. Configure the Groq API key

Create a file called `.env` in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

The application uses `openai/gpt-oss-120b` in `rag_pipeline.py`. Confirm that this model is available to your Groq account; change the `model` value there if needed.

**Security:** Never publish your `.env` file or API key. Add `.env`, `.venv/`, `pdfs/`, and `vectorstore/` to your `.gitignore` as appropriate.

## Run the application

In a terminal with your virtual environment activated, run:

```bash
streamlit run frontend.py
```

If the `streamlit` command is not found, try:

```bash
python -m streamlit run frontend.py
```

Open **http://localhost:8501** if it does not open automatically.

## How to use

1. Upload a text-based PDF through the Streamlit file uploader.
2. Wait for the **PDF processed successfully!** confirmation.
3. Type a question about the uploaded document.
4. Click **Ask AI Lawyer** to see the response.

**Example questions**

- "What are the main rights discussed in this document?"
- "Summarize the obligations described in Section 2."
- "What does the PDF say about legal remedies?"

The prompt instructs the model to respond with **"I don't know based on the provided document."** if the supplied context lacks an answer. This is an instruction, not a guarantee against hallucinations.

## How it works

```text
Uploaded PDF
    |
    v
PDFPlumberLoader (extract text per page)
    |
    v
RecursiveCharacterTextSplitter (1,000-character chunks; 200 overlap)
    |
    v
OllamaEmbeddings (nomic-embed-text)
    |
    v
FAISS vector index (in memory)
    |
    v
User question -> Top 4 similar chunks
    |
    v
LangChain prompt + retrieved context
    |
    v
Groq-hosted LLM (openai/gpt-oss-120b)
    |
    v
Answer in Streamlit
```

### Main files

| File | Responsibility |
| --- | --- |
| `frontend.py` | Uploads PDFs, builds a FAISS index, retrieves matching chunks, and renders the UI. |
| `vector_db.py` | Provides upload, parsing, splitting, embedding, and optional index-saving utilities. |
| `rag_pipeline.py` | Initializes the Groq model, builds the context/prompt, and invokes the model. |

## Current implementation notes

- **Reprocessing:** Streamlit reruns the script on widget interactions. In the current code, uploading a PDF causes parsing and embedding to run again on later reruns, including when asking a question. Consider `st.session_state` or caching to reuse the index.
- **Index persistence:** `frontend.py` builds the FAISS index in memory. The `save_local("vectorstore/db_faiss")` call is only in the standalone test block of `vector_db.py`; the main app does not persist the index.
- **Retrieval:** The app performs FAISS similarity search with `k=4` and passes the matching chunk text to the LLM. It does not currently display page citations.
- **Scanned PDFs:** Image-only/scanned PDFs may need OCR; the current PDFPlumber-based code does not add an OCR step.
- **Data handling:** Uploaded PDFs are saved under `pdfs/` on the machine hosting Streamlit. Retrieved text is sent to the Groq API for generation; do not upload confidential material without checking your data-handling requirements.

## Troubleshooting

**`Connection refused` or Ollama connection errors:** Make sure Ollama is installed and running, and that `nomic-embed-text` has been downloaded.

**Groq authentication errors:** Check `GROQ_API_KEY` in `.env`, then restart Streamlit. Confirm that the API key is active.

**Groq model errors:** Verify the configured model is offered to your Groq account or select another supported model in `rag_pipeline.py`.

**`ModuleNotFoundError`:** Activate the intended virtual environment and rerun `python -m pip install -r requirements.txt`.

**Poor results with a scanned PDF:** Use a text-based PDF or add an OCR pipeline before chunking.

## License

Add a `LICENSE` file if you plan to distribute this project publicly.
