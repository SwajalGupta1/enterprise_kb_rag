# ingestion/loaders.py
import pdfplumber
import pandas as pd
import requests
from bs4 import BeautifulSoup
from uuid import uuid4
from datetime import datetime

def load_direct_text(text: str):
    return {
        "id": str(uuid4()),
        "source": "direct_text",
        "content": text,
        "metadata": {"type": "direct", "ingested_at": datetime.utcnow().isoformat()}
    }

def load_txt(uploaded_file):
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    return {
        "id": str(uuid4()),
        "source": uploaded_file.name,
        "content": raw,
        "metadata": {"type": "txt", "ingested_at": datetime.utcnow().isoformat()}
    }

def load_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text += f"[PAGE {i}]\n" + (page.extract_text() or "") + "\n"

    return {
        "id": str(uuid4()),
        "source": uploaded_file.name,
        "content": text,
        "metadata": {"type": "pdf", "ingested_at": datetime.utcnow().isoformat()}
    }

def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    rows = []
    for i, row in df.iterrows():
        rows.append(f"ROW {i}: " + "; ".join(f"{col}: {row[col]}" for col in df.columns))
    text = "\n".join(rows)

    return {
        "id": str(uuid4()),
        "source": uploaded_file.name,
        "content": text,
        "metadata": {"type": "csv", "columns": list(df.columns),
                     "rows": len(df), "ingested_at": datetime.utcnow().isoformat()}
    }

def load_url(url: str):
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    text = "\n".join([l.strip() for l in soup.get_text().splitlines() if l.strip()])

    return {
        "id": str(uuid4()),
        "source": url,
        "content": text,
        "metadata": {"type": "url", "ingested_at": datetime.utcnow().isoformat()}
    }
