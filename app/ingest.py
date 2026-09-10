import logging
import os
from typing import List, Tuple

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from pypdf import PdfReader

from config import (
    DATA_DIR,
)

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