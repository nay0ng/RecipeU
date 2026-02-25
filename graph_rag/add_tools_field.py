"""
add_tools_field.py

레시피 JSON에 cooking_tools 필드 추가
step 텍스트에서 규칙 기반으로 조리 도구 추출
"""

import json
import re
from pathlib import Path
from collections import Counter

# =============================================
# 조리 도구 사전
# 일반명 → 정규화된 이름으로 매핑
# =============================================
TOOL_DICT = {
    # 밥솥
    "밥솥": "밥솥",
    "전기밥솥": "밥솥",
    "압력밥솥": "밥솥",

    # 전자레인지
    "전자레인지": "전자레인지",

    # 오븐
    "오븐": "오븐",
    "오븐기": "오븐",
    "토스터오븐": "오븐",

    # 에어프라이어
    "에어프라이어": "에어프라이어",
    "에어후라이어": "에어프라이어",

    # 찜기
    "찜기": "찜기",
    "찜솥": "찜기",

    # 믹서기
    "믹서기": "믹서기",
    "믹서": "믹서기",
    "블렌더": "믹서기",
    "핸드블렌더": "믹서기",

    # 착즙기
    "착즙기": "착즙기",
    "착즙": "착즙기",

    # 커피머신
    "커피머신": "커피머신",
    "커피메이커": "커피머신",
    "에스프레소머신": "커피머신",

    # 토스트기
    "토스트기": "토스트기",
    "토스터": "토스트기",

    # 와플메이커
    "와플메이커": "와플메이커",
    "와플기": "와플메이커",
}

# 긴 이름 먼저 매칭하기 위해 길이 내림차순 정렬
SORTED_TOOLS = sorted(TOOL_DICT.keys(), key=len, reverse=True)


# =============================================
# 도구 추출 함수
# =============================================
def extract_tools(steps: list[str]) -> list[str]:
    """
    step 리스트에서 조리 도구 추출
    중복 제거 후 정규화된 이름으로 반환
    """
    found_tools = set()
    full_text = " ".join(steps)

    for tool_keyword in SORTED_TOOLS:
        if tool_keyword in full_text:
            normalized = TOOL_DICT[tool_keyword]
            found_tools.add(normalized)

    return sorted(list(found_tools))


# =============================================
# 메인
# =============================================
def main():
    INPUT_PATH = "../recipe_data/recipes_180000.json"   # 입력 파일
    OUTPUT_PATH = "../recipe_data/recipes_180000_with_tools.json"   # 출력 파일

    print(f"파일 로딩 중: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)
    print(f"총 {len(recipes):,}개 레시피 로드 완료")

    # 도구 추출
    tool_counter = Counter()
    no_tool_count = 0

    for recipe in recipes:
        steps = recipe.get("steps", [])
        tools = extract_tools(steps)
        recipe["cooking_tools"] = tools  # 필드 추가

        if tools:
            tool_counter.update(tools)
        else:
            no_tool_count += 1

    # 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)

    # 통계 출력
    print(f"\n✅ 저장 완료: {OUTPUT_PATH}")
    print(f"\n📊 통계")
    print(f"  도구 없음:  {no_tool_count:,}개 ({no_tool_count/len(recipes)*100:.1f}%)")
    print(f"  도구 있음:  {len(recipes)-no_tool_count:,}개 ({(len(recipes)-no_tool_count)/len(recipes)*100:.1f}%)")
    print(f"\n🔧 많이 등장한 도구 Top 15:")
    for tool, count in tool_counter.most_common(15):
        print(f"  {tool:<15} {count:,}개")

    # 샘플 출력
    print(f"\n📋 샘플 확인 (첫 3개):")
    for recipe in recipes[:3]:
        print(f"  [{recipe['title'][:20]}] → {recipe['cooking_tools']}")


if __name__ == "__main__":
    main()