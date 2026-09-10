from typing import List

import ollama

from config import EMBED_MODEL

def embed_texts(texts : List[str]) -> List[List[float]]:
    if not texts:
        return []
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return list(response.embeddings) # type:ignore

def embed_query(text: str):
    return embed_texts([text])[0]