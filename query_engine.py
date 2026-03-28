"""
Chat engine with per-user conversation memory for the Gilded Age RAG bot.

Key functions:
  get_index()            — load ChromaDB index once at startup
  get_chat_engine(index) — create a fresh CondensePlusContextChatEngine
                           with its own memory buffer (one per user session)
"""
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_storage"
COLLECTION = "gilded_age"
SIMILARITY_TOP_K = 8

SYSTEM_PROMPT = """\
You are a culinary research assistant for food styling work on HBO's The Gilded Age. \
Your knowledge base covers Gilded Age culinary history, period menus and cookbooks, \
food costing and yields, food safety regulations, and production resources for the show.

Prioritize information from the retrieved source documents. If the sources contain \
a relevant answer, use them and cite them. If the sources are insufficient, \
supplement with your general knowledge — but make clear what comes from the KB \
versus general knowledge.

Calibrate your response length to the question:
- For simple lookups (yields, temperatures, dates, counts): answer in 1–2 sentences.
- For descriptive questions (a person's style, a room, a dish, a technique): \
write a full, vivid, detailed response — do not truncate.
- For list questions: use bullet points.
Always cite your sources at the end using the format: _Source: [document name]_\
"""

CONTEXT_PROMPT = (
    "You are a culinary research assistant for HBO's The Gilded Age production.\n\n"
    "Use the source excerpts below as your primary reference. "
    "Cite them if they answer the question. "
    "Supplement with general knowledge when sources are insufficient, noting what comes from each.\n\n"
    "Calibrate length to the question: short factual questions get direct 1-2 sentence answers; "
    "descriptive questions about people, places, dishes, or techniques deserve full, vivid detail.\n\n"
    "Always end with: _Source: [document name(s)]_\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
)


def get_index() -> VectorStoreIndex:
    """Load the ChromaDB index. Call once at startup and share across sessions."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No ChromaDB storage found at {CHROMA_DIR}. Run ingest.py first."
        )

    Settings.llm         = OpenAI(model="gpt-4o", temperature=0.2, system_prompt=SYSTEM_PROMPT)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    chroma_client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_ctx       = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
    )


def get_chat_engine(index: VectorStoreIndex) -> CondensePlusContextChatEngine:
    """Create a new chat engine with a fresh memory buffer for one user session."""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=SIMILARITY_TOP_K)
    memory    = ChatMemoryBuffer.from_defaults(token_limit=4096)

    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        context_prompt=CONTEXT_PROMPT,
        verbose=False,
    )


if __name__ == "__main__":
    print("Loading index…")
    _index = get_index()
    engine = get_chat_engine(_index)
    print("Gilded Age assistant ready (chat mode). Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            response = engine.chat(q)
            print(f"\nAssistant: {response}\n")
