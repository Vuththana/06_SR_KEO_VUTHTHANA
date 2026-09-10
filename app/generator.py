import ollama
from typing import List

from config import GEN_MODEL
from config import SYSTEM_PROMPT
from retriever import RetrievedChunk


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        context_block = "(no relevant context was found)"
    else:
        formatted_chunks = [
            f"--- Context Block {i+1} ---\n{c['text']}"
            for i, c in enumerate(chunks)
        ]
        context_block = "\n\n".join(formatted_chunks)

    return f"""You are an IT support assistant. Answer the user's question clearly and naturally using ONLY the provided documentation.

    ### Guidelines:
    1. Direct Answer: Start directly with 1-2 concise sentences answering the query.
    2. Details: Provide 2-4 supporting bullet points explaining key steps, prerequisites, or reasons found in the documentation.
    3. Strict Grounding: Do not invent facts, swap roles (e.g., IT vs user), or assume details not present in the text.
        
    ### Documentation Context:
    {context_block}

    ### User Question:
    {query}

    ### Response:"""

def generate_answer(query: str, chunks:List[RetrievedChunk]) -> str:
    prompt = build_prompt(query, chunks)

    response  = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role" : "system" , "content" : SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.message.content
