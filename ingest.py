"""
Ingest all documents from the KB folder into a persistent ChromaDB vector store.
Run this once (or re-run to rebuild the index).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
KB_DIR      = Path(__file__).parent / "KB"
CHROMA_DIR  = Path(__file__).parent / "chroma_storage"
COLLECTION  = "gilded_age"

Settings.llm         = OpenAI(model="gpt-4o", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size    = 1024
Settings.chunk_overlap = 128

# ── Load documents ─────────────────────────────────────────────────────────────
print("Loading documents from KB…")
reader = SimpleDirectoryReader(
    input_dir=str(KB_DIR),
    recursive=False,
    required_exts=[".pdf", ".txt", ".md"],
    # Image-only PDFs are excluded — their text is extracted by ocr_pdfs.py
    # and saved as .txt files which are ingested instead
    exclude=[
        "delmonico_menu.pdf",
        "grover cleveland menu 1891.pdf",
        "What's on the Menu - About.pdf",
        "Grover Cleveland NYT state dinner.pdf",
    ],
)
documents = reader.load_data()
file_count = len(set(d.metadata.get("file_name", "?") for d in documents))
print(f"  Loaded {len(documents)} document nodes from {file_count} files")

# ── ChromaDB setup ─────────────────────────────────────────────────────────────
CHROMA_DIR.mkdir(exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Delete existing collection to allow clean rebuild
try:
    chroma_client.delete_collection(COLLECTION)
    print(f"  Cleared existing collection '{COLLECTION}'")
except Exception:
    pass

chroma_collection = chroma_client.get_or_create_collection(COLLECTION)
vector_store  = ChromaVectorStore(chroma_collection=chroma_collection)
storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)

# ── Build & persist index ──────────────────────────────────────────────────────
print("Building vector index (this will take several minutes)…")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_ctx,
    show_progress=True,
)
print(f"\nIndex saved to {CHROMA_DIR}")
