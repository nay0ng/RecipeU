import os
import re
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import APIRouter, HTTPException, Query
from typing import List
from datetime import datetime, timedelta
from features.ranking.schemas import RecipeDetail, RecipePreview, RankingResponse

router = APIRouter()

# MongoDB 연결 (MONGO_URI 미설정 시 랭킹 기능 비활성화로 graceful 처리)
MONGODB_URL = os.getenv("MONGO_URI", "")
DATABASE_NAME = os.getenv("DATABASE_NAME", "recipe_db")

client = None
db = None

if MONGODB_URL:
    try:
        client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=3000)
        db = client[DATABASE_NAME]
    except Exception as _e:
        print(f"⚠️  MongoDB 연결 실패 (랭킹 기능 비활성): {_e}")

RANKING_CACHE = {
    "today": None,
    "updated_at": None,
}


async def load_today_ranking_cache():
    """오늘 랭킹을 미리 메모리에 로드 (순서 보존 버전)"""
    if db is None:
        print("⚠️  MongoDB 없음 - 랭킹 캐시 로딩 스킵")
        return

    now = datetime.now()
    
    if now.hour < 7:
        now = now - timedelta(days=1)
    
    today_kst = now.strftime("%Y-%m-%d")

    ranking_data = await db.ranking_id.find_one(
        {
            "date_kst": today_kst,
            "source": "10000recipes",
        },
        {"recipe_ids": 1, "_id": 0},
        sort=[("created_at_kst", -1)],
    )

    if not ranking_data:
        print("❌ 랭킹 데이터 없음")
        return

    recipe_ids = ranking_data.get("recipe_ids", [])

    if not recipe_ids:
        print("❌ recipe_ids 비어있음")
        return

    recipes_raw = await db.recipes.find(
        {"recipe_id": {"$in": recipe_ids}},
        {"recipe_id": 1, "title": 1, "author": 1, "image": 1, "_id": 0}
    ).to_list(length=200)

    if not recipes_raw:
        print("❌ recipes 컬렉션 조회 실패")
        return

    recipe_map = {r["recipe_id"]: r for r in recipes_raw}

    # 🚀 Pydantic 거치지 않고 바로 dict로 저장
    previews = [
        {
            "recipe_id": r["recipe_id"],
            "title": r.get("title", ""),
            "author": r.get("author", ""),
            "image": r.get("image", ""),
        }
        for rid in recipe_ids
        if (r := recipe_map.get(rid))
    ]

    RANKING_CACHE["today"] = {
        "date_kst": today_kst,
        "recipes": previews,  # 🚀 이미 dict
        "total_count": len(previews),
    }

    RANKING_CACHE["updated_at"] = now

    print(f"✅ 랭킹 캐시 완료 ({len(previews)}개)")


import time

@router.get("/today")
async def get_today_ranking(limit: int = Query(100, ge=1, le=100)):
    if db is None:
        return {"date_kst": "", "recipes": [], "total_count": 0}

    start = time.time()

    print(f"🔍 캐시 확인: {RANKING_CACHE['today'] is not None}")

    if RANKING_CACHE["today"]:
        data = RANKING_CACHE["today"]
        
        result = {
            "date_kst": data["date_kst"],
            "recipes": data["recipes"][:limit],
            "total_count": data["total_count"],
        }
        
        elapsed = time.time() - start
        print(f"✅ 캐시에서 반환: {len(data['recipes'])}개 - {elapsed*1000:.2f}ms 걸림")
        
        return result

    await load_today_ranking_cache()

    if not RANKING_CACHE["today"]:
        raise HTTPException(404, "No ranking data")

    data = RANKING_CACHE["today"]
    
    result = {
        "date_kst": data["date_kst"],
        "recipes": data["recipes"][:limit],
        "total_count": data["total_count"],
    }
    
    elapsed = time.time() - start
    print(f"✅ 로딩 후 반환: {elapsed*1000:.2f}ms 걸림")

    return result

@router.get("/{date_kst}", response_model=RankingResponse)
async def get_ranking_by_date(
    date_kst: str,
    limit: int = Query(100, ge=1, le=100),
):
    if db is None:
        raise HTTPException(503, "랭킹 서비스를 사용할 수 없습니다 (MongoDB 미연결)")

    try:
        datetime.strptime(date_kst, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format")

    ranking_data = await db.ranking_id.find_one(
        {
            "date_kst": date_kst,
            "source": "10000recipes",
        },
        sort=[("created_at_kst", -1)],
    )

    if not ranking_data:
        raise HTTPException(404, "No ranking data")

    recipe_ids = ranking_data.get("recipe_ids", [])

    recipes = await db.recipes.find({"recipe_id": {"$in": recipe_ids}}).to_list(
        length=200
    )

    previews = [
        RecipePreview(
            recipe_id=r["recipe_id"],
            title=r["title"],
            author=r.get("author", ""),
            image=r.get("image", ""),
        )
        for r in recipes
    ]

    return RankingResponse(
        date_kst=date_kst,
        recipes=previews[:limit],
        total_count=len(previews),
    )


@router.get("/search", response_model=List[RecipePreview])
async def search_recipes(
    keyword: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
):
    if db is None:
        return []

    cursor = db.recipes.find(
        {
            "$or": [
                {"title": {"$regex": keyword, "$options": "i"}},
                {"ingredients.name": {"$regex": keyword, "$options": "i"}},
            ]
        }
    ).limit(limit)

    recipes = []

    async for r in cursor:
        recipes.append(
            RecipePreview(
                recipe_id=r["recipe_id"],
                title=r["title"],
                author=r.get("author", ""),
                image=r.get("image", ""),
            )
        )

    return recipes


# ===============================
# 레시피 상세 (단건 조회)
# ===============================


@router.get("/recipes/{recipe_id}", response_model=RecipeDetail)
async def get_recipe_detail(recipe_id: str):
    if db is None:
        raise HTTPException(503, "랭킹 서비스를 사용할 수 없습니다 (MongoDB 미연결)")

    recipe = await db.recipes.find_one({"recipe_id": recipe_id})

    if not recipe:
        raise HTTPException(404, "Recipe not found")

    steps = recipe.get("steps", [])
    cleaned_steps = []
    for step in steps:
        # "11. " 같은 패턴 제거
        cleaned_step = re.sub(r'^\d+\.\s*', '', step)
        cleaned_steps.append(cleaned_step)
        
    return RecipeDetail(
        recipe_id=recipe["recipe_id"],
        title=recipe["title"],
        author=recipe.get("author", ""),
        image=recipe.get("image", ""),
        intro=recipe.get("intro", ""),
        portion=recipe.get("portion", ""),
        cook_time=recipe.get("cook_time", ""),
        level=recipe.get("level", ""),
        detail_url=recipe.get("detail_url", ""),
        ingredients=recipe.get("ingredients", []),
        steps=cleaned_steps,
    )