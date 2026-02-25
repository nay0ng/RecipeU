import time
from typing import List
from lib.chunking import chunk_documents
from langchain_community.vectorstores import Milvus

from lib.mongo_utils import get_mongo_collections, get_unembedded_recipes, mark_embedded

MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
COLLECTION_NAME = "recipe_docs"
MODEL_NAME = "bge-m3"


def embedding_and_upsert(documents, embeddings, recipes_col, sleep_per_doc=1.0):

    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": MILVUS_HOST,
            "port": MILVUS_PORT,
        },
        drop_old=False,
        auto_id=True,
    )

    for doc in documents:
        vectorstore.add_texts(
            texts=[doc.page_content],
            metadatas=[doc.metadata],
        )

        recipe_id = doc.metadata.get("recipe_id")
        if recipe_id:
            mark_embedded(recipes_col, recipe_id, MODEL_NAME)

        time.sleep(sleep_per_doc)
