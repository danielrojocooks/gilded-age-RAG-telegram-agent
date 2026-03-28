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

QA_TEMPLATE = PromptTemplate(
    "You are a culinary research assistant for HBO's The Gilded Age production.\n\n"
    "Use the source excerpts below as your primary reference. "
    "If they fully answer the question, cite them. "
    "If they are incomplete, supplement with general knowledge and note what comes from each.\n\n"
    "Calibrate length to the question: short factual questions get direct 1-2 sentence answers; "
    "descriptive questions about people, places, dishes, or techniques deserve full, vivid detail.\n\n"
    "Always end with: _Source: [document name(s)]_\n\n"
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
