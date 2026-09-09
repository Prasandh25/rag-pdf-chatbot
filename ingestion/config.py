"""
Shared config for embedding model + vector store location.
Keeping this in one place avoids embedder.py and vector_store.py
silently drifting out of sync on model choice or persist path.
"""

import os

# Small, fast, well-regarded general-purpose embedding model.
# Runs locally, no API key needed.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Where Chroma persists its data on disk.
PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "vectorstore"
)

COLLECTION_NAME = "pdf_chunks"


def get_embedding_function():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)