# ingestion/processors.py
def clean_text(text: str):
    t = text.replace("\r", "\n")
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    return "\n".join(lines)
