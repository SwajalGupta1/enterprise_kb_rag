# querying/generator.py


from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt, model="llama-3.3-70b-versatile"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2
        )

        # Correct extraction for Groq API
        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {e}"



# ----------------------
# RAG Prompt Builder
# ----------------------
def build_rag_prompt(query, retrieved_docs):
    context_blocks = []

    for i, d in enumerate(retrieved_docs, 1):
        safe_chunk = d["chunk"][:1200]
        context_blocks.append(
            f"[Source {i}] ({d['source']} - chunk {d['metadata']['chunk_index']})\n{safe_chunk}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are an intelligent RAG assistant.
Use only the following knowledge base context to answer:

-------------------- CONTEXT --------------------
{context_text}
-------------------------------------------------

User query: {query}

Rules:
- Only answer from context.
- If answer not found, say: "The provided knowledge base does not contain this answer."
- Cite like [Source 1], etc.
"""
    return prompt.strip()
