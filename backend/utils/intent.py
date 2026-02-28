# backend/utils/intent.py
"""
의도 감지 유틸
"""
from typing import List
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage


class Intent:
    # 조리 모드 의도
    NEXT = "next_step"
    PREV = "prev_step"
    SUB_ING = "substitute_ingredient"
    SUB_TOOL = "substitute_tool"
    FAILURE = "failure"

    # 채팅 모드 의도
    RECIPE_SEARCH = "recipe_search"  # 새로운 레시피 검색/추천
    RECIPE_MODIFY = "recipe_modify"  # 기존 레시피 수정 요청
    COOKING_QUESTION = "cooking_question"  # 요리 관련 일반 질문
    NOT_COOKING = "not_cooking"  # 요리 무관

    UNKNOWN = "unknown"


def detect_intent(text: str) -> str:
    """조리 모드 의도 감지 (LLM 기반)"""

    prompt = f"""조리 중 의도 분류:

입력: {text}

분류:
1. NEXT - 다음 단계로
2. PREV - 이전 단계로
3. SUB_ING - 재료 대체
4. SUB_TOOL - 도구 대체
5. FAILURE - 조리 실패
6. UNKNOWN - 기타

**출력 형식: 분류 키워드만 1개 출력**
출력:"""

    try:
        llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=20)
        result = llm.invoke([HumanMessage(content=prompt)])
        decision = result.content.strip().upper()

        if "NEXT" in decision:
            return Intent.NEXT
        elif "PREV" in decision:
            return Intent.PREV
        elif "SUB_ING" in decision:
            return Intent.SUB_ING
        elif "SUB_TOOL" in decision:
            return Intent.SUB_TOOL
        elif "FAILURE" in decision:
            return Intent.FAILURE
        else:
            return Intent.UNKNOWN
    except Exception as e:
        print(f"[Intent] 조리 모드 LLM 분류 실패: {e}")
        # Fallback
        t = text.lower()
        if any(k in t for k in ["다음", "넘겨"]):
            return Intent.NEXT
        elif any(k in t for k in ["이전", "뒤로"]):
            return Intent.PREV
        elif any(k in t for k in ["탔", "망했"]):
            return Intent.FAILURE
        return Intent.UNKNOWN


def extract_constraints(text: str) -> List[str]:
    """제약 조건 추출"""
    constraints = []
    content = text.replace(" ", "").lower()

    if any(k in content for k in ["초보", "쉬운", "간단"]):
        constraints.append("쉬운")
    if any(k in content for k in ["빠른", "빨리"]):
        constraints.append("빠른")
    if any(k in content for k in ["건강", "다이어트"]):
        constraints.append("저칼로리")

    return constraints


def extract_allergy_dislike(text: str, chat_history: list = None) -> dict:
    """알러지/비선호 음식 추출

    Args:
        text: 사용자 입력 텍스트
        chat_history: 대화 히스토리 (레시피가 있는지 확인용)

    Returns:
        {
            "type": "allergy" | "dislike" | None,
            "items": ["재료1", "재료2", ...],
            "original_text": "원본 텍스트"
        }
    """

    # 대화에 레시피가 있으면 "빼고"는 RECIPE_MODIFY 의도이므로 None 반환
    has_recipe = False
    if chat_history:
        for msg in reversed(chat_history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "재료" in content and ("⏱️" in content or "📊" in content):
                    has_recipe = True
                    break

    # 레시피가 있고 "빼고/제외/싫어" 같은 수정 키워드가 있으면 → RECIPE_MODIFY 의도
    # "싫어해/싫어" 포함: 레시피 보는 중이면 해당 재료 제거 후 재생성
    modify_keywords = ["빼고", "뺴고", "빼줘", "뺴줘", "제외", "말고", "대신", "싫어", "안먹어"]
    text_normalized = text.replace(" ", "")

    # 오타 허용 (ㅐ ↔ ㅏ 혼동)
    has_modify_keyword = any(
        keyword in text_normalized or
        keyword.replace("빼", "뺴") in text_normalized or
        keyword.replace("뺴", "빼") in text_normalized
        for keyword in modify_keywords
    )

    if has_recipe and has_modify_keyword:
        print(f"[AllergyDetect] 레시피 존재 + 수정 키워드 감지 → RECIPE_MODIFY 의도로 판단, 알러지 감지 스킵")
        return {"type": None, "items": [], "original_text": text}

    prompt = f"""사용자가 **직접적으로** 알러지나 비선호 음식을 언급했는지 분석:

입력: "{text}"

**중요: 단순 메뉴 언급이나 레시피 수정 요청은 NONE!**
- "고수덮밥 먹을까" → NONE (메뉴 질문일 뿐)
- "후추 빼고" → NONE (레시피 수정 요청)
- "나 고수 싫어해" → DISLIKE (개인 선호 명시)
- "새우 알러지 있어" → ALLERGY (알러지 명시)

**분류:**
1. ALLERGY - 알러지/알레르기 **명시적 진술**
   예: "나 땅콩 알러지 있어", "새우 못먹어", "우유 먹으면 배아파"

2. DISLIKE - 싫어하는 음식 **명시적 진술**
   예: "나 파 싫어해", "고수 안먹어"

3. NONE - 알러지/비선호 아님
   예: "~~ 먹을까", "~~ 어때", "~~ 만들어줘"

**출력 형식: 아래 형식으로만 출력**
타입: ALLERGY 또는 DISLIKE 또는 NONE
재료: 재료1, 재료2 (없으면 "없음")"""

    try:
        llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=50)
        result = llm.invoke([HumanMessage(content=prompt)])
        response = result.content.strip()

        print(f"[AllergyDetect] 입력: {text}")
        print(f"[AllergyDetect] LLM 응답: {response}")

        # ✅ 응답 품질 체크 (공백 비율로 판단)
        # 정상 응답이면 공백이 10% 이상 있어야 함
        space_count = response.count(' ') + response.count('\n')
        total_chars = len(response)
        space_ratio = space_count / total_chars if total_chars > 0 else 0

        print(f"[AllergyDetect] 응답 품질: 공백 비율 {space_ratio:.2%} ({space_count}/{total_chars}자)")

        # 공백 비율이 너무 낮거나, 필수 키워드가 없으면 품질 불량
        has_required_keywords = any(kw in response for kw in ["타입:", "재료:", "ALLERGY", "DISLIKE", "NONE"])

        if space_ratio < 0.05 or not has_required_keywords:
            print(f"[AllergyDetect] LLM 응답 품질 불량 → 키워드 기반으로 폴백")
            raise ValueError("LLM 응답 품질 불량")

        # 파싱
        detected_type = None
        items = []

        if "ALLERGY" in response.upper():
            detected_type = "allergy"
        elif "DISLIKE" in response.upper():
            detected_type = "dislike"
        elif "NONE" in response.upper():
            return {"type": None, "items": [], "original_text": text}

        # 재료 추출
        if "재료:" in response:
            items_text = response.split("재료:")[-1].strip()
            if items_text and items_text != "없음":
                items = [item.strip() for item in items_text.split(",")]
                items = [item for item in items if item and len(item) > 0]

        print(f"[AllergyDetect] 타입: {detected_type}, 재료: {items}")

        if detected_type and items:
            return {
                "type": detected_type,
                "items": items,
                "original_text": text
            }
        else:
            return {"type": None, "items": [], "original_text": text}

    except Exception as e:
        print(f"[AllergyDetect] LLM 추출 실패: {e}")
        print(f"[AllergyDetect] 키워드 기반 폴백 실행")

        # Fallback: 룰베이스 (키워드 기반)
        text_lower = text.lower()

        # 오타 교정: 자주 발생하는 오타 패턴
        typo_corrections = {
            "뺴고": "빼고",
            "뺴줘": "빼줘",
            "뺴": "빼",
            "싷어": "싫어",
            "안머거": "안먹어",
            "제와": "제외",
        }

        for typo, correct in typo_corrections.items():
            if typo in text_lower:
                text_lower = text_lower.replace(typo, correct)
                print(f"[AllergyDetect] 오타 교정: {typo} → {correct}")

        # 알러지 키워드
        allergy_keywords = ["알러지", "알레르기", "못먹어", "먹으면", "배아파", "탈나"]
        is_allergy = any(k in text_lower for k in allergy_keywords)

        # 비선호 키워드
        dislike_keywords = ["싫어", "안먹어", "빼줘", "빼고", "제외"]
        is_dislike = any(k in text_lower for k in dislike_keywords)

        if is_allergy or is_dislike:
            # 간단한 재료 추출 (키워드 앞의 단어)
            detected_type = "allergy" if is_allergy else "dislike"

            # 재료 추출 시도 (단순 패턴 매칭)
            import re
            # "재료명 + 키워드" 패턴 찾기
            items = []

            # 교정된 텍스트와 원본 텍스트 모두에서 추출 시도
            for keyword in (allergy_keywords if is_allergy else dislike_keywords):
                # 교정된 텍스트에서 추출
                pattern = r'([가-힣]+)\s*' + re.escape(keyword)
                matches = re.findall(pattern, text_lower)
                items.extend(matches)

                # 원본 텍스트에서도 추출 (공백 제거)
                text_no_space = text.replace(" ", "")
                matches = re.findall(pattern, text_no_space.lower())
                items.extend(matches)

            # 중복 제거
            items = list(set(items))

            print(f"[AllergyDetect] 키워드 기반 추출: 타입={detected_type}, 재료={items}")

            return {
                "type": detected_type,
                "items": items,  # 키워드로 추출한 재료
                "original_text": text
            }

        return {"type": None, "items": [], "original_text": text}


def extract_ingredients_from_modification(text: str, mod_type: str = "remove") -> dict:
    """수정 요청에서 재료명 추출

    Args:
        text: 사용자 수정 요청
        mod_type: 수정 타입 ("remove", "replace", "add")

    Returns:
        {
            "remove": ["재료1"],  # 제거할 재료
            "add": ["재료2"]      # 추가할 재료
        }

    예:
        "아 근데 참치 없어 빼줘" (remove) → {"remove": ["참치"], "add": []}
        "돼지고기 말고 참치 넣어줘" (replace) → {"remove": ["돼지고기"], "add": ["참치"]}
    """

    # Replace 타입일 때는 구분하여 추출
    if mod_type == "replace":
        prompt = f"""사용자의 레시피 수정 요청을 분석하세요.

입력: "{text}"

**규칙:**
1. "A 말고 B" 패턴에서 A는 제거, B는 추가
2. 재료명만 추출 (조사 "로/을/를/은/는/으로" 등 반드시 제거, 동사 제거)
3. 예: "참치로" → "참치", "돼지고기를" → "돼지고기"

**출력 형식:**
제거: 재료명
추가: 재료명

예시:
입력: "돼지고기 말고 참치로 바꿔줘"
제거: 돼지고기
추가: 참치

출력:"""

        try:
            llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=50)
            result = llm.invoke([HumanMessage(content=prompt)])
            response = result.content.strip()

            print(f"[IngredientExtract] Replace 타입 - 입력: {text}")
            print(f"[IngredientExtract] LLM 응답: {response}")

            remove_items = []
            add_items = []

            # 파싱 (콜론 앞 공백 허용: "제거:" / "제거 :" 모두 매칭)
            for line in response.split('\n'):
                line_stripped = line.strip()
                if re.match(r'^제거\s*:', line_stripped):
                    items_text = re.sub(r'^제거\s*:\s*', '', line_stripped)
                    if items_text and items_text != "없음":
                        remove_items = [item.strip() for item in items_text.split(",")]
                elif re.match(r'^추가\s*:', line_stripped):
                    items_text = re.sub(r'^추가\s*:\s*', '', line_stripped)
                    if items_text and items_text != "없음":
                        add_items = [item.strip() for item in items_text.split(",")]

            print(f"[IngredientExtract] 제거: {remove_items}, 추가: {add_items}")
            return {"remove": remove_items, "add": add_items}

        except Exception as e:
            print(f"[IngredientExtract] Replace LLM 추출 실패: {e}")

            # Fallback: "A 말고 B" 패턴 매칭
            import re
            pattern = r'([가-힣]+)\s*말고\s*([가-힣]+)'
            match = re.search(pattern, text)
            if match:
                remove_items = [match.group(1)]
                add_items = [match.group(2)]
                print(f"[IngredientExtract] Fallback - 제거: {remove_items}, 추가: {add_items}")
                return {"remove": remove_items, "add": add_items}

            return {"remove": [], "add": []}

    # Remove/Add 타입일 때는 기존 로직
    else:
        prompt = f"""사용자의 레시피 수정 요청에서 음식 재료명만 추출하세요.

입력: "{text}"

**규칙:**
1. 음식 재료명만 추출 (조사, 동사, 장소 제거)
2. 여러 재료면 모두 추출
3. 재료가 없으면 "없음"

**출력 형식: 재료명만 쉼표로 구분**
예시:
- "참치 빼줘" → 참치
- "집에 간장이 없어" → 간장
- "오이 집에 없어 빼줘" → 오이
- "딸기, 블루베리 추가해줘" → 딸기, 블루베리
- "알려줘" → 없음

재료:"""

        try:
            llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=50)
            result = llm.invoke([HumanMessage(content=prompt)])
            response = result.content.strip()

            print(f"[IngredientExtract] {mod_type} 타입 - 입력: {text}")
            print(f"[IngredientExtract] LLM 응답: {response}")

            if "없음" in response or not response:
                print(f"[IngredientExtract] LLM이 재료 추출 실패 → Fallback 시도")
                raise ValueError("LLM이 재료를 추출하지 못함")

            # 쉼표로 분리
            ingredients = [item.strip() for item in response.split(",")]
            ingredients = [item for item in ingredients if item and len(item) > 0]

            print(f"[IngredientExtract] 추출된 재료: {ingredients}")

            if mod_type == "remove":
                return {"remove": ingredients, "add": []}
            else:  # add
                return {"remove": [], "add": ingredients}

        except Exception as e:
            print(f"[IngredientExtract] LLM 추출 실패: {e}")

            # Fallback: 간단한 패턴 매칭
            import re

            # 키워드 앞의 명사 추출 (조사 포함 패턴)
            keywords = ["빼", "제거", "없이", "말고", "없어", "없는", "없다", "대신", "바꿔", "교체", "빼줘"]
            ingredients = []

            for keyword in keywords:
                # 패턴: 명사(+조사) + 공백 + 키워드
                # 예: "간장이 없어", "참치를 빼", "돼지고기 말고"
                pattern = r'([가-힣]+?)(?:이|가|을|를|은|는|도|만|에|에서|으로|로)?\s*' + re.escape(keyword)
                matches = re.findall(pattern, text)
                ingredients.extend(matches)

            # 장소/컨텍스트 명사 제외 (재료가 아닌 단어들)
            location_words = ["집", "냉장고", "부엌", "주방", "마트", "편의점", "가게", "슈퍼", "어제", "오늘", "내일"]
            ingredients = [ing for ing in ingredients if ing not in location_words]

            # 아무것도 못 찾았으면 문장 맨 앞의 명사 시도 (빼/없어 키워드가 있을 때만)
            if not ingredients and any(kw in text for kw in ["빼", "제거", "없어", "없는", "없다"]):
                # 문장 맨 앞의 명사 추출 (공백이나 조사 전까지)
                match = re.match(r'^([가-힣]{2,})', text)
                if match:
                    first_word = match.group(1)
                    if first_word not in location_words:
                        ingredients.append(first_word)
                        print(f"[IngredientExtract] 문장 맨 앞 명사 추출: {first_word}")

            # 중복 제거 및 필터링
            ingredients = list(set([ing for ing in ingredients if ing and len(ing) >= 2]))
            print(f"[IngredientExtract] Fallback 추출: {ingredients}")

            if mod_type == "remove":
                return {"remove": ingredients, "add": []}
            else:
                return {"remove": [], "add": ingredients}


def detect_chat_intent(text: str, chat_history: list = None) -> str:
    """LLM 기반 채팅 의도 감지 - 레시피 검색/수정/일반질문/무관 구분"""

    # 대화 히스토리에서 최근 레시피 확인 (assistant 메시지 중 레시피 찾을 때까지)
    has_recipe = False
    if chat_history:
        # 최근 메시지부터 역순으로 검색, 레시피 찾을 때까지 계속
        for msg in reversed(chat_history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                # "재료:" 또는 "**재료:**"가 있으면 레시피로 판단
                if "재료" in content and ("⏱️" in content or "📊" in content):
                    has_recipe = True
                    break  # 레시피 발견하면 중단

    # LLM 프롬프트 (간결하고 명확하게)
    prompt = f"""의도 분류:

입력: "{text}"
레시피: {"Y" if has_recipe else "N"}

**중요: 음식/요리 키워드 없으면 무조건 NOT_COOKING**
**개인정보 보호: 이름, 전화번호, 주소, 이메일 등 개인정보(PII) 포함 입력은 무조건 NOT_COOKING**

분류:
1. NOT_COOKING (음식/요리 무관 또는 개인정보 포함)
   예: "날씨", "영화", "여행", "제주도", "운동", "음악"
   예: "내 이름은 홍길동", "010-1234-5678", "서울시 강남구"
   → 음식/요리 키워드 없거나 개인정보 포함 시 무조건 이것

2. RECIPE_MODIFY (레시피=Y + 재료 수정/제거)
   예: "딸기 빼줘", "A 말고 B", "더 맵게"
   **중요: "집에 [재료] 없어", "[재료] 없는데", "[재료]가 없다" → RECIPE_MODIFY**
   예: "집에 오이가 없어", "간장이 없는데", "참치 없다"

3. RECIPE_SEARCH (음식/요리 관련, 레시피=N)
   예: "김치찌개", "케이크 먹을까", "빵"

4. COOKING_QUESTION (요리 지식)
   예: "보관법", "칼로리", "쯔유 대신 간장?"

**출력 형식: 분류 키워드만 1개 출력 (설명 없이)**
출력:"""

    try:
        llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=20)
        result = llm.invoke([HumanMessage(content=prompt)])
        decision = result.content.strip().upper().replace(" ", "")

        print(f"[Intent] 입력: {text}")
        print(f"[Intent] 레시피 존재: {has_recipe}")
        print(f"[Intent] LLM 응답: {decision}")

        # 응답 파싱
        # "MOD"도 인식 (LLM이 "RECIPE_MOD"로 응답할 수 있음)
        if "RECIPE_MODIFY" in decision or "RECIPE_MOD" in decision or "MODIFY" in decision or ("MOD" in decision and "Y" in decision):
            print(f"[Intent] → RECIPE_MODIFY")
            return Intent.RECIPE_MODIFY
        elif "NOT_COOKING" in decision or "NOTCOOKING" in decision:
            print(f"[Intent] → NOT_COOKING")
            return Intent.NOT_COOKING
        elif "COOKING_QUESTION" in decision or "COOKINGQUESTION" in decision or "QUESTION" in decision:
            print(f"[Intent] → COOKING_QUESTION")
            return Intent.COOKING_QUESTION
        elif "RECIPE_SEARCH" in decision or "RECIPESEARCH" in decision or "SEARCH" in decision:
            print(f"[Intent] → RECIPE_SEARCH")
            return Intent.RECIPE_SEARCH
        else:
            # LLM 응답이 명확하지 않으면 기본값: RECIPE_SEARCH
            print(f"[Intent] → RECIPE_SEARCH (LLM 응답 불명확, 기본값)")
            return Intent.RECIPE_SEARCH

    except Exception as e:
        print(f"[Intent] LLM 의도 분류 실패: {e}")
        # Fallback: 최소한의 룰베이스
        text_lower = text.lower()

        # 요리 무관 키워드 우선 확인
        not_cooking_keywords = ["영화", "날씨", "여행", "제주", "부산", "서울", "운동", "음악", "게임", "드라마", "뉴스", "정치", "경제"]
        if any(k in text_lower for k in not_cooking_keywords):
            print(f"[Intent] Fallback → NOT_COOKING (요리 무관 키워드 감지)")
            return Intent.NOT_COOKING

        # 레시피 수정 키워드 (명확함)
        # "싫어해/안먹어" 포함: 레시피 보는 중이면 해당 재료 제거 후 재생성
        modify_keywords = ["말고", "대신", "바꿔", "교체", "추가", "빼고", "빼줘", "제거", "없이", "더", "덜", "없어", "없는", "없다", "싫어", "안먹어"]
        if has_recipe and any(k in text_lower for k in modify_keywords):
            print(f"[Intent] Fallback → RECIPE_MODIFY")
            return Intent.RECIPE_MODIFY

        # 기본값: RECIPE_SEARCH (보수적)
        print(f"[Intent] Fallback → RECIPE_SEARCH (기본값)")
        return Intent.RECIPE_SEARCH