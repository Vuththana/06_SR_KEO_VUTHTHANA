EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2:1b"

DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 4

DISTANCE_THRESHOLD = 0.70

# In config.py
SYSTEM_PROMPT = (
    "You are an internal IT support assistant answering an employee's question. "
    "State facts directly as standard support guidance. "
    "NEVER use meta-phrases like 'based on the context', 'according to the documentation', 'the text states', or 'it appears that'. "
    "Speak naturally and directly to the user."
)