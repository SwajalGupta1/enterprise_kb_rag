import os
import faiss
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Get absolute path of project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "onnx_bge")

MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

class VectorIndexer:
    def __init__(self):
        # Load tokenizer from local folder
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )

        # Load ONNX model with correct absolute path
        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        # embedding dimension 384 for BGE small
        self.index = faiss.IndexFlatL2(384)
        self.store = []

    def embed(self, texts):
        # Tokenize text
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np"
        )

        # If token_type_ids are missing (they usually are), add zeros
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = np.zeros_like(encoded["input_ids"])

        ort_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"]
        }

        # ONNX forward pass
        outputs = self.session.run(None, ort_inputs)

        # Your model ONLY returns last_hidden_state
        last_hidden = outputs[0]     # shape: (batch, seq_len, 384)

        # ⭐ Apply mean pooling across seq_len → final shape (batch, 384)
        embeddings = last_hidden.mean(axis=1).astype("float32")

        return embeddings




    def add_documents(self, docs):
        print("DEBUG: Received docs:", len(docs))
        texts = [d["chunk"] for d in docs]
        print("DEBUG: First chunk:", texts[0] if texts else "None")

        embeddings = self.embed(texts)
        print("DEBUG: Embeddings shape:", embeddings.shape)

        self.index.add(embeddings.astype("float32"))
        print("DEBUG: FAISS size after add:", self.index.ntotal)

        self.store.extend(docs)
        print("DEBUG: Store size:", len(self.store))


    def search(self, query, top_k=5):
        print("DEBUG: FAISS ntotal:", self.index.ntotal)

        if self.index.ntotal == 0:
            print("DEBUG: Index is EMPTY")
            return []

        query_emb = self.embed([query])
        print("DEBUG: Query emb shape:", query_emb.shape)

        distances, indices = self.index.search(query_emb.astype("float32"), top_k)
        print("DEBUG: Returned indices:", indices)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.store):
                results.append(self.store[idx])
        print("DEBUG: Results count:", len(results))
        return results

if __name__ == "__main__":
    idx = VectorIndexer()
    test = idx.embed(["I am Swajal Gupta"])
    print("Embedding shape:", test.shape)
    print("Embedding sample:", test[0][:10])
