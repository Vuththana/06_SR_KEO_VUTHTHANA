
from typing import List, TypedDict

from config import TOP_K
from embeddings import embed_query
from ingest import get_collection

class RetrievedChunk(TypedDict):
    text: str
    source: str
    chunk_index: int
    distance: float

def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"] #type:ignore
    )

    chunks: List[RetrievedChunk] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": distance,
            } #type:ignore
        )
    return chunks