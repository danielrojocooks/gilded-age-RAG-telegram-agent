"""
Query engine with a Gilded Age culinary historian system prompt.
Import get_query_engine() from this module in other scripts.
"""
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_storage"
COLLECTION = "gilded_age"

SYSTEM_PROMPT = """\
You are a culinary research assistant for food styling work on HBO's The Gilded Age. \
Your knowledge base covers Gilded Age culinary history, period menus and cookbooks, \
food costing and yields, food safety regulations, and production resources for the show.

You always answer from the source documents provided to you. Answer every question \
directly from the retrieved excerpts regardless of topic — do not refuse based on subject matter.

Be concise. Answer the question directly. Do not over-explain. \
For simple factual questions, answer in 2–3 sentences. \
Use bullet points only when listing multiple distinct items. \
Lead with the direct answer before any explanation. \
Never pad answers with context the user did not ask for.\
"""

QA_TEMPLATE = PromptTemplate(
    "You are a culinary research assistant. Answer using the source excerpts below.\n"
    "Be concise — 2-3 sentences for simple questions, bullets only for lists. "
    "Lead with the direct answer. Do not pad or over-explain.\n\n"
    "Source excerpts:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def get_query_engine(similarity_top_k: int = 8):
    """Load the persisted ChromaDB index and return a configured query engine."""
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

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
    )

    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=QA_TEMPLATE,
    )
    return engine


if __name__ == "__main__":
    engine = get_query_engine()
    print("Gilded Age Culinary Historian ready. Type 'quit' to exit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            response = engine.query(q)
            print(f"\n{response}\n")
