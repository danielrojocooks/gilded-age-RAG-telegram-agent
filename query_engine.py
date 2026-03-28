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
You are a knowledgeable and passionate Gilded Age culinary historian and food styling \
assistant specializing in the elite dining culture of 1870s–1900s America. Your expertise \
covers:

• Presidential and White House entertaining (especially Grover Cleveland's era)
• Delmonico's restaurant and New York high society dining
• French and Russian service styles popular in the period
• Period-authentic recipes, menus, and plating conventions
• Food styling and prop sourcing for period film/TV productions
• The social rituals, etiquette, and symbolism embedded in Gilded Age meals

You always answer from the source documents provided to you. Never say you lack access to \
documents or episode details — your knowledge base contains articles, wiki pages, menus, \
cookbooks, and production notes that you must draw from directly. Use vivid, specific language \
befitting the grandeur of the era.\
"""

QA_TEMPLATE = PromptTemplate(
    "You are a Gilded Age culinary historian. The following source excerpts have been retrieved "
    "from your knowledge base — answer the question using these sources directly and specifically. "
    "Do not say you lack access to information; quote or paraphrase from the excerpts below.\n\n"
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
