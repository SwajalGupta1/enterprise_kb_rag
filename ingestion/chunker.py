# ingestion/chunker.py
from uuid import uuid4
from datetime import datetime

def chunk_text(text: str, max_words=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start = end - overlap

    return chunks

def create_chunks(raw_doc):
    cleaned = raw_doc["content"]
    chunks = chunk_text(cleaned)

    final_docs = []
    for i, chunk in enumerate(chunks):
        final_docs.append({
            "id": str(uuid4()),
            "parent_id": raw_doc["id"],
            "chunk": chunk,
            "source": raw_doc["source"],
            "metadata": {**raw_doc["metadata"], "chunk_index": i}
        })
    return final_docs
