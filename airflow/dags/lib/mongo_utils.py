from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING

MONGO_URI = "mongodb://root:RootPassword123@mongodb:27017/admin"
DB_NAME = "recipe_db"
RECIPE_COL = "recipes"


# MongoDB
def get_mongo_collections():
    mongo = MongoClient(MONGO_URI)
    db = mongo[DB_NAME]
    recipes = db[RECIPE_COL]

    recipes.create_index([("recipe_id", ASCENDING)], unique=True)

    return recipes


def get_unembedded_recipes(recipes, limit=150):
    """
    아직 임베딩 안 된 레시피 조회
    """
    return (
        recipes.find({"embedded": {"$ne": True}}).sort("registered_at", -1).limit(limit)
    )


def mark_embedded(recipes, recipe_id, model_name):
    """
    임베딩 완료 표시
    """
    return recipes.update_one(
        {"recipe_id": recipe_id},
        {
            "$set": {
                "embedded": True,
                "embedded_at": datetime.now(timezone.utc),
                "embedding_model": model_name,
            }
        },
    )
