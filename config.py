EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2:1b"

DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 4

SYSTEM_PROMPT = {
    "You are a helpful assistant that answers question using ONLY the provided context below."
    "If the answer is not in the context answer with \"I don't have enough information to answer yet.\" "
    "Cite the source file name(s) you used."
}