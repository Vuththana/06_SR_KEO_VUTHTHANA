from typing import List, TypedDict

from config import TOP_K, DISTANCE_THRESHOLD
from embeddings import embed_query
from vector_store import ChromaStore


class RetrievedChunk(TypedDict):
    text: str
    source: str
    chunk_index: int
    distance: float


def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    chroma_db = ChromaStore()
    collection = chroma_db.get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],  # type:ignore
    )

    chunks: List[RetrievedChunk] = []
    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    for text, meta, distance in zip(documents, metadatas, distances):
        # Guard against None values in metadata
        safe_meta = meta or {}
        if distance > DISTANCE_THRESHOLD:
            continue

        chunks.append(
            {
                "text": text,
                "source": safe_meta.get("source", "unknown"),
                "chunk_index": safe_meta.get("chunk_index", -1),
                "distance": distance,
            }
        )
    return chunks