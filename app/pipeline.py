from retriever import retrieve, RetrievedChunk
from generator import generate_answer
from ingest import load_documents
from embeddings import embed_texts
from vector_store import ChromaStore
from typing import List

chroma_db = ChromaStore()

def initialize_vector_store():
    collection = chroma_db.get_collection()

    if collection.count() > 0:
        return

    print("Indexing documents into ChromaDB...")
    documents = load_documents()

    for filename, text in documents:
        embeddings = embed_texts([text])  # Returns [[float, float, ...]]

        chroma_db.add(
            ids=[f"{filename}_0"],
            documents=[text],
            embeddings=embeddings,
            metadata=[{"source": filename, "chunk_index": 0}],
        )
    print("Indexing complete.")

def pipeline(prompt: str) -> str:
    initialize_vector_store()

    chunks: List[RetrievedChunk] = retrieve(query=prompt, top_k=5)

    # No chunk retrieved return fallback
    if not chunks:
        return "I don't have enough information for it, sorry!"
    # Print retrieved chunks
    else:
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            distance = chunk.get("distance", 0.0)

            print(f"\n[Chunk {i}] Source: {source} | Distance: {distance:.4f}")
            print("-" * 60)

    print("=" * 60 + "\n")

    response = generate_answer(query=prompt, chunks=chunks)
    return response