import time
import logging
import pytz
import requests

from venv import logger
from bs4 import BeautifulSoup
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

from lib.page_crawling import crawl_recipe_detail

RANKING_URL = "https://www.10000recipe.com/ranking/home_new.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

MONGO_URI = "mongodb://root:RootPassword123@mongodb:27017/admin"
DB_NAME = "recipe_db"
DB_RANKING_COLLECTION = "ranking_id"
DB_RECIPE_COLLECTION = "recipes"
session = requests.Session()
session.headers.update(HEADERS)


# 랭킹 레시피 -> recipe_id 수집
def get_recipe_ids_by_ranking():
    print("start")
    res = session.get(RANKING_URL, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    recipe_ids = []
    for a in soup.select("a.common_sp_link"):
        href = a.get("href", "")
        if href.startswith("/recipe/"):
            recipe_ids.append(href.split("/")[-1])
    return list(dict.fromkeys(recipe_ids))


def save_ranking_to_mongo(recipe_ids):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[DB_RANKING_COLLECTION]

    KST = pytz.timezone("Asia/Seoul")

    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    now_kst = now_utc.astimezone(KST)

    doc = {
        "date_kst": now_kst.strftime("%Y-%m-%d"),
        "date_utc": now_utc.strftime("%Y-%m-%d"),
        "source": "10000recipes",
        "type": "ranking",
        "url": RANKING_URL,
        "recipe_ids": recipe_ids,
        "count": len(recipe_ids),
        "created_at_kst": now_kst,
        "created_at_utc": now_utc,
    }

    try:
        col.insert_one(doc)
    except DuplicateKeyError:
        logger.info("오늘 랭킹 이미 저장됨 -> skip")


# 레시피 정보가 있는 db에서 id 가져오기
def get_existing_recipe_ids(col, recipe_ids):
    cursor = col.find({"recipe_id": {"$in": recipe_ids}}, {"recipe_id": 1, "_id": 0})
    return {doc["recipe_id"] for doc in cursor}


def process_ranking_to_recipes():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    recipe_col = db[DB_RECIPE_COLLECTION]

    # 랭킹 수집
    ranking_recipe_ids = get_recipe_ids_by_ranking()

    # 랭킹 데이터 저장
    save_ranking_to_mongo(ranking_recipe_ids)

    if not ranking_recipe_ids:
        logger.error("랭킹 recipe_id 수집 실패")
        return

    existing_recipe_ids = get_existing_recipe_ids(recipe_col, ranking_recipe_ids)

    new_recipe_ids = [
        rid for rid in ranking_recipe_ids if rid not in existing_recipe_ids
    ]
    logger.info(
        f"랭킹 recipe_id {len(ranking_recipe_ids)}개 중 "
        f"신규 레시피 {len(new_recipe_ids)}개"
    )

    for recipe_id in new_recipe_ids:
        try:
            recipe_doc = crawl_recipe_detail(recipe_id)
            recipe_col.insert_one(recipe_doc)
            logger.info(f"레시피 저장 성공: recipe_id={recipe_id}")
        except Exception as e:
            logger.exception(f"레시피 크롤링/저장 실패: recipe_id={recipe_id}")
