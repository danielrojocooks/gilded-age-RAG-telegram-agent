"""
Ingest all documents from the KB folder into a persistent ChromaDB vector store.
Processes files in small batches to stay within Railway's 512MB memory limit.
Run this once (or re-run to rebuild the index).
"""
import gc
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
KB_DIR     = Path(__file__).parent / "KB"
CHROMA_DIR = Path(__file__).parent / "chroma_storage"
COLLECTION = "gilded_age"
BATCH_SIZE = 2  # files per batch — large text files OOM at 5 on Railway 512MB

EXCLUDE = {
    "delmonico_menu.pdf",
    "grover cleveland menu 1891.pdf",
    "What's on the Menu - About.pdf",
    "Grover Cleveland NYT state dinner.pdf",
    # Replaced by book_of_yields_extracted.txt (table-aware extraction)
    "The-Book-of-Yields-Accuracy-in-Food-Costing-and-Purchasing.pdf",
}

Settings.llm         = OpenAI(model="gpt-4o", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size    = 1024
Settings.chunk_overlap = 128

# ── Gather file list ───────────────────────────────────────────────────────────
all_files = [
    f for f in KB_DIR.iterdir()
    if f.suffix.lower() in {".pdf", ".txt", ".md"}
    and f.name not in EXCLUDE
]
all_files.sort(key=lambda f: f.stat().st_size)  # small files first
print(f"Found {len(all_files)} files to ingest")

# ── ChromaDB setup ─────────────────────────────────────────────────────────────
CHROMA_DIR.mkdir(exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    chroma_client.delete_collection(COLLECTION)
    print(f"  Cleared existing collection '{COLLECTION}'")
except Exception:
    pass

chroma_collection = chroma_client.get_or_create_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)

# ── Ingest in batches ──────────────────────────────────────────────────────────
total_docs = 0
for batch_start in range(0, len(all_files), BATCH_SIZE):
    batch = all_files[batch_start : batch_start + BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE
    names = ", ".join(f.name for f in batch)
    print(f"\nBatch {batch_num}/{total_batches}: {names}")

    reader = SimpleDirectoryReader(input_files=[str(f) for f in batch])
    documents = reader.load_data()
    total_docs += len(documents)
    print(f"  {len(documents)} chunks — embedding…")

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        show_progress=False,
    )

    # Free memory between batches
    del documents
    gc.collect()

print(f"\nDone. {total_docs} total chunks indexed into {CHROMA_DIR}")
