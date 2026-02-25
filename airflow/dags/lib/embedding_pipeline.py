from lib.embed_and_upsert import embedding_and_upsert
from langchain_community.embeddings import ClovaXEmbeddings
from lib.chunking import chunk_documents
from lib.mongo_utils import get_mongo_collections, get_unembedded_recipes
from lib.recipe_to_doc import recipe_to_document

MODEL_NAME = "bge-m3"


def run_embedding_pipeline():
    recipes = get_mongo_collections()

    raw_recipes = list(get_unembedded_recipes(recipes, limit=150))

    docs = [recipe_to_document(r) for r in raw_recipes]
    docs = chunk_documents(docs)

    embeddings = ClovaXEmbeddings(
        model=MODEL_NAME,
    )

    embedding_and_upsert(docs, embeddings, recipes)
