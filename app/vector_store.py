import chromadb

from config import (
    COLLECTION_NAME,
    CHROMA_DB_DIR,
    TOP_K
)

class ChromaStore:
    def __init__(self, chroma_db_dir: str = CHROMA_DB_DIR, collection_name: str = COLLECTION_NAME) -> chromadb:
        self.client = chromadb.PersistentClient(path=chroma_db_dir)

        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, documents, embeddings, metadata=None):
        return self.collection.add(ids=ids, documents=documents, embeddings=embeddings,metadatas=metadata)

    def search(self, query_embeddings, n_results=TOP_K):
        return self.collection.search(query_embeddings, n_results)

    def get_collection(self, collection_name: str = COLLECTION_NAME):
        return self.client.get_collection(name=collection_name)