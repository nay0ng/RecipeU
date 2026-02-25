"""
update_tools_property.py

Tool 노드 대신 Recipe 속성으로 cooking_tools 저장
- 기존 데이터 삭제 없이 속성만 추가
- 관계 40만개 제한 우회
"""

import json
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

JSON_FILE_PATH = "../recipe_data/recipes_180000_with_tools.json"
BATCH_SIZE = 500
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

UPDATE_QUERY = """
UNWIND $batch AS item
MATCH (r:Recipe {id: item.recipe_id})
SET r.cooking_tools = item.cooking_tools
"""

def main():
    print(f"JSON 로딩 중: {JSON_FILE_PATH}")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"총 {len(recipes):,}개 레시피 로드 완료")

    # 배치 준비 (cooking_tools 있든 없든 전부)
    batches = []
    buf = []
    for recipe in recipes:
        buf.append({
            "recipe_id":     recipe["recipe_id"],
            "cooking_tools": recipe.get("cooking_tools", [])
        })
        if len(buf) >= BATCH_SIZE:
            batches.append(buf)
            buf = []
    if buf:
        batches.append(buf)

    print(f"총 {len(batches)}개 배치")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    start = time.time()
    try:
        for i, batch in enumerate(batches):
            with driver.session() as session:
                session.run(UPDATE_QUERY, batch=batch)
            if (i + 1) % 20 == 0:
                print(f"  {(i+1)*BATCH_SIZE:,}/{len(recipes):,} ({time.time()-start:.1f}s)")

        print(f"\n✅ 완료! ({time.time()-start:.1f}s)")

        # 검증
        with driver.session() as session:
            result = session.run("""
                MATCH (r:Recipe)
                WHERE size(r.cooking_tools) > 0
                RETURN count(r) AS cnt
            """).single()
            print(f"  cooking_tools 있는 레시피: {result['cnt']:,}개")

            result2 = session.run("""
                MATCH (r:Recipe)
                WHERE "에어프라이어" IN r.cooking_tools
                RETURN count(r) AS cnt
            """).single()
            print(f"  에어프라이어 레시피: {result2['cnt']:,}개")

    finally:
        driver.close()

if __name__ == "__main__":
    main()