import logging
import os
from typing import List, Tuple

import chromadb
from chromadb.config import Settings

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
from pypdf import PdfReader

from config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
)
from embeddings import embed_texts

def _read_text(path:str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_pdf(path:str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_documents(data_dir: str = DATA_DIR) -> List[Tuple[str, str]]:
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in ("txt", "md"):
            text = _read_text(path)
        elif ext == "pdf":
            text = _read_pdf(path)
        else:
            continue
        if text.strip():
            documents.append((filename, text))
        return documents

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=_CHROMA_SETTINGS)
    return client.get_or_create_collection(name=COLLECTION_NAME)