"""
Neo4j AuraDB 레시피 데이터 배치 임포트 스크립트
18만 개 데이터 고속 처리용

변경사항:
  - Step 노드 제거 → Recipe.steps 속성으로 통합
  - Tool 노드 추가 (REQUIRES_TOOL 관계)
  - recipes_180000_with_tools.json 사용
"""

import json
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================
# 설정
# =============================================
JSON_FILE_PATH = "../recipe_data/recipes_180000_with_tools.json"
BATCH_SIZE = 500
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# =============================================
# 인덱스 생성
# =============================================
def create_indexes(driver):
    print("인덱스 생성 중...")
    with driver.session() as session:
        session.run("CREATE INDEX recipe_id        IF NOT EXISTS FOR (r:Recipe)     ON (r.id)")
        session.run("CREATE INDEX ingredient_name  IF NOT EXISTS FOR (i:Ingredient) ON (i.name)")
        session.run("CREATE INDEX tool_name        IF NOT EXISTS FOR (t:Tool)       ON (t.name)")
        session.run("CREATE INDEX recipe_level     IF NOT EXISTS FOR (r:Recipe)     ON (r.level)")
        session.run("CREATE INDEX recipe_cook_time IF NOT EXISTS FOR (r:Recipe)     ON (r.cook_time)")
    print("인덱스 생성 완료!")


# =============================================
# Cypher 쿼리
# =============================================

# 레시피 노드 - steps를 속성으로 저장
RECIPE_BATCH_QUERY = """
UNWIND $batch AS recipe
MERGE (r:Recipe {id: recipe.recipe_id})
SET r.title         = recipe.title,
    r.intro         = recipe.intro,
    r.author        = recipe.author,
    r.detail_url    = recipe.detail_url,
    r.image         = recipe.image,
    r.portion       = recipe.portion,
    r.cook_time     = recipe.cook_time,
    r.level         = recipe.level,
    r.registered_at = recipe.registered_at,
    r.steps         = recipe.steps_text
"""

# 재료 관계
INGREDIENT_BATCH_QUERY = """
UNWIND $batch AS item
MERGE (r:Recipe {id: item.recipe_id})
MERGE (i:Ingredient {name: item.ingredient_name})
MERGE (r)-[rel:CONTAINS]->(i)
SET rel.amount = item.amount
"""

# 조리 도구 관계
TOOL_BATCH_QUERY = """
UNWIND $batch AS item
MERGE (r:Recipe {id: item.recipe_id})
MERGE (t:Tool {name: item.tool_name})
MERGE (r)-[:REQUIRES_TOOL]->(t)
"""


# =============================================
# 배치 삽입 함수
# =============================================
def insert_recipe_batch(driver, batch):
    with driver.session() as session:
        session.run(RECIPE_BATCH_QUERY, batch=batch)

def insert_ingredient_batch(driver, batch):
    with driver.session() as session:
        session.run(INGREDIENT_BATCH_QUERY, batch=batch)

def insert_tool_batch(driver, batch):
    with driver.session() as session:
        session.run(TOOL_BATCH_QUERY, batch=batch)


# =============================================
# 전처리: JSON → 배치 리스트 변환
# =============================================
def prepare_batches(recipes, batch_size):
    recipe_batches    = []
    ingredient_batches = []
    tool_batches      = []

    recipe_buf, ingredient_buf, tool_buf = [], [], []

    for recipe in recipes:
        # steps 리스트 → 하나의 텍스트로 합치기
        steps_list = recipe.get("steps", [])
        steps_text = "\n".join(steps_list)

        recipe_buf.append({
            **recipe,
            "steps_text": steps_text
        })
        if len(recipe_buf) >= batch_size:
            recipe_batches.append(recipe_buf)
            recipe_buf = []

        # 재료 관계
        for ing in recipe.get("ingredients", []):
            name = ing.get("name", "").strip()
            if not name:
                continue
            ingredient_buf.append({
                "recipe_id":       recipe["recipe_id"],
                "ingredient_name": name,
                "amount":          ing.get("amount", "")
            })
            if len(ingredient_buf) >= batch_size:
                ingredient_batches.append(ingredient_buf)
                ingredient_buf = []

        # 조리 도구 관계 (있는 것만)
        for tool in recipe.get("cooking_tools", []):
            if not tool:
                continue
            tool_buf.append({
                "recipe_id": recipe["recipe_id"],
                "tool_name": tool
            })
            if len(tool_buf) >= batch_size:
                tool_batches.append(tool_buf)
                tool_buf = []

    # 남은 버퍼 처리
    if recipe_buf:      recipe_batches.append(recipe_buf)
    if ingredient_buf:  ingredient_batches.append(ingredient_buf)
    if tool_buf:        tool_batches.append(tool_buf)

    return recipe_batches, ingredient_batches, tool_batches


# =============================================
# 메인 실행
# =============================================
def main():
    print(f"JSON 파일 로딩 중: {JSON_FILE_PATH}")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"총 {len(recipes):,}개 레시피 로드 완료")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # 1. 인덱스 생성
        create_indexes(driver)

        # 2. 배치 준비
        print(f"\n배치 분할 중 (배치 크기: {BATCH_SIZE})...")
        recipe_batches, ingredient_batches, tool_batches = prepare_batches(recipes, BATCH_SIZE)
        print(f"레시피 배치:  {len(recipe_batches)}개")
        print(f"재료 배치:    {len(ingredient_batches)}개")
        print(f"도구 배치:    {len(tool_batches)}개")

        total_start = time.time()

        # 3. 레시피 노드 삽입
        print(f"\n[1/3] 레시피 노드 삽입 중...")
        start = time.time()
        for i, batch in enumerate(recipe_batches):
            insert_recipe_batch(driver, batch)
            if (i + 1) % 10 == 0:
                print(f"  {(i+1)*BATCH_SIZE:,}/{len(recipes):,} ({time.time()-start:.1f}s)")
        print(f"  완료! ({time.time()-start:.1f}s)")

        # 4. 재료 관계 삽입
        print(f"\n[2/3] 재료 관계 삽입 중...")
        start = time.time()
        for i, batch in enumerate(ingredient_batches):
            insert_ingredient_batch(driver, batch)
            if (i + 1) % 20 == 0:
                print(f"  배치 {i+1}/{len(ingredient_batches)} ({time.time()-start:.1f}s)")
        print(f"  완료! ({time.time()-start:.1f}s)")

        # 5. 도구 관계 삽입
        print(f"\n[3/3] 조리 도구 관계 삽입 중...")
        start = time.time()
        for i, batch in enumerate(tool_batches):
            insert_tool_batch(driver, batch)
            if (i + 1) % 20 == 0:
                print(f"  배치 {i+1}/{len(tool_batches)} ({time.time()-start:.1f}s)")
        print(f"  완료! ({time.time()-start:.1f}s)")

        total_elapsed = time.time() - total_start
        print(f"\n✅ 전체 완료! 총 소요 시간: {total_elapsed/60:.1f}분")

        # 6. 검증
        print("\n데이터 검증 중...")
        with driver.session() as session:
            recipe_count     = session.run("MATCH (r:Recipe) RETURN count(r) AS cnt").single()["cnt"]
            ingredient_count = session.run("MATCH (i:Ingredient) RETURN count(i) AS cnt").single()["cnt"]
            tool_count       = session.run("MATCH (t:Tool) RETURN count(t) AS cnt").single()["cnt"]
            contains_count   = session.run("MATCH ()-[r:CONTAINS]->() RETURN count(r) AS cnt").single()["cnt"]
            tool_rel_count   = session.run("MATCH ()-[r:REQUIRES_TOOL]->() RETURN count(r) AS cnt").single()["cnt"]

        print(f"  레시피 노드:        {recipe_count:,}개")
        print(f"  재료 노드:          {ingredient_count:,}개")
        print(f"  도구 노드:          {tool_count:,}개")
        print(f"  CONTAINS 관계:      {contains_count:,}개")
        print(f"  REQUIRES_TOOL 관계: {tool_rel_count:,}개")

    finally:
        driver.close()


if __name__ == "__main__":
    main()