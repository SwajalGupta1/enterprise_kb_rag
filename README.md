

# 🤖 Multi-Source Retrieval-Augmented AI Assistant (Groq + FAISS + Streamlit)

OmniRAG is an advanced **Retrieval-Augmented Generation (RAG)** system that ingests PDFs, CSVs, text files, web URLs, or raw text and instantly builds a searchable knowledge base. It generates embeddings using **Groq’s LLaMA 3.2 Embedding model**, performs semantic search with **FAISS**, and produces accurate grounded answers using **Groq LLaMA models** — all inside a clean, intuitive Streamlit interface.

---

## 🚀 Features

- Multi-source ingestion: **PDF, TXT, CSV, Website URLs, Direct Text**
- Automatic text extraction, cleaning & preprocessing
- Smart chunking with metadata tracking
- Embeddings via **Groq LLaMA 3.2 Embedding Model**
- Fast vector search using **FAISS**
- RAG-based answer generation using **Groq LLaMA 3.x models**
- Streamlit UI with:
  - 📥 Ingestion Tab  
  - ❓ Querying Tab
- Fully deployable on **Streamlit Cloud**
- No local model downloads required — cloud-based LLM & embeddings

---

## 🧠 Tech Stack

- **Python 3.13**
- **Streamlit**
- **Groq API (Embeddings + LLMs)**
- **FAISS**
- **PyPDF2**
- **BeautifulSoup4**
- **pandas / numpy**
- **dotenv / Streamlit Secrets**
- **Requests**
- **GitHub + Streamlit Cloud**

---

## 📁 Project Structure



- enterprise_kb_rag/
- │
- ├── app/
- │ └── streamlit_app.py
- │
- ├── ingestion/
- │ ├── loaders.py
- │ ├── processors.py
- │ ├── chunker.py
- │ └── indexer.py
- │
- ├── querying/
- │ ├── retriever.py
- │ └── generator.py
- │
- ├── requirements.txt
- └── README.md
---

## ⚙️ How OmniRAG Works

1. User uploads files or enters text/URL  
2. System extracts raw text from PDF/CSV/TXT/Web  
3. Text is cleaned, normalized, and chunked  
4. Each chunk gets embedded using **Groq Embeddings API**  
5. FAISS indexes vectors for semantic search  
6. User asks a question  
7. Query is embedded → top-k chunks retrieved  
8. A structured RAG prompt is generated with sources  
9. **Groq LLaMA model** produces the final answer  

---

## Streamlit URL
https://enterprisekbrag-kyqyus86xqqtlh3jdknssz.streamlit.app/


