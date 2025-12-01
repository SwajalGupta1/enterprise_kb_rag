# ingestion/indexer.py

import os
import numpy as np
import requests
import faiss

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "llama-3.2-embedding-1b"

class VectorIndexer:
    def __init__(self):
        self.index = None
        self.store = []

    def embed(self, texts):
        """Generate embeddings from Groq API"""
        url = "https://api.groq.com/openai/v1/embeddings"

        payload = {
            "model": EMBED_MODEL,
            "input": texts
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            vectors = [item["embedding"] for item in data["data"]]
            return np.array(vectors).astype("float32")

        except Exception as e:
            print("Embedding Error:", e)
            return np.zeros((len(texts), 384)).astype("float32")

    def add_documents(self, chunks):
        """Add chunk embeddings into FAISS index"""
        texts = [c["chunk"] for c in chunks]
        embeddings = self.embed(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        # store original metadata
        for i, c in enumerate(chunks):
            self.store.append({
                "chunk": c["chunk"],
                "source": c["source"],
                "metadata": c["metadata"]
            })

    def search(self, query, top_k=5):
        """Search top k results"""
        query_emb = self.embed([query])

        D, I = self.index.search(query_emb, top_k)

        results = []
        for idx in I[0]:
            if idx < len(self.store):
                results.append(self.store[idx])

        return results
