# 🔎 Enterprise Knowledge Base RAG System  
### Powered by FAISS • Groq LLMs • Streamlit

This project is a **full end-to-end RAG (Retrieval Augmented Generation) system** that can ingest data from multiple sources (PDF, TXT, CSV, URL, direct text), process and chunk the content, index it using FAISS, and answer user queries using a fast, high-accuracy LLM hosted on **Groq**.

The final application is deployed on **Streamlit Cloud** and works fully with API keys via Streamlit Secrets.

---

## 🚀 Features

- 📥 **Multi-source ingestion**
  - Direct text input
  - PDF files
  - TXT files
  - CSV files
  - Website URL extraction (BeautifulSoup)

- ✂️ **Automatic text cleaning & chunking**

- ⚡ **Vector Embeddings (384-dimension)**
  - Using BGE-small ONNX model

- 🔍 **FAISS vector search**
  - Fast nearest-neighbor lookup
  - Persistent inside Streamlit session

- 🤖 **LLM-powered answer generation**
  - Groq LLMs (`llama-3.3-70b-versatile`)
  - Accurate contextual responses
  - Proper RAG citation format

- 🧱 **Modular code structure**
  - Clean separation of loaders, processing, chunking, indexing, querying

- ☁️ **One-click deployment on Streamlit Cloud**

---

## 📂 Folder Structure

enterprse_kb/
│
├── app/
│ └── streamlit_app.py
│
├── ingestion/
│ ├── loaders.py
│ ├── processors.py
│ ├── chunker.py
│ └── indexer.py
│
├── querying/
│ ├── retriever.py
│ └── generator.py
│
├── requirements.txt
└── README.md




