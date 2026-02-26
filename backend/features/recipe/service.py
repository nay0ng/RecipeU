# backend/features/recipe/service.py
"""
Recipe 비즈니스 로직
"""
import re
import json
from typing import List, Dict, Any
from toon_format import decode as toon_decode
from .prompts import RECIPE_QUERY_EXTRACTION_PROMPT, RECIPE_GENERATION_PROMPT, RECIPE_DETAIL_EXPANSION_PROMPT


# ─────────────────────────────────────────────
# 토큰 사용량 추적 헬퍼 함수
# ─────────────────────────────────────────────
# 요청별 토큰 누적 (요청당 초기화됨)
_token_accumulator: dict = {"prompt": 0, "completion": 0, "total": 0}
# 단계별 토큰 정보 저장 (단계명 -> {prompt, completion, total})
_step_tokens: dict = {}
# 단계별 시간 추적 (단계명 -> 시간(ms))
_step_timings: dict = {}
def print_token_usage(response, context_name: str = "LLM"):
    """LLM 응답에서 실제 토큰 사용량 출력 (개선 버전)"""
    print(f"\n{'='*60}")
    print(f"[{context_name}] HCX API 토큰 사용량 (실측)")
    print(f"{'='*60}")

    # ✅ 개선: usage_metadata 우선 확인 (LangChain 표준)
    usage = None
    source = ""

    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        source = "usage_metadata"
    elif hasattr(response, 'response_metadata'):
        usage = response.response_metadata.get('token_usage')
        source = "response_metadata.token_usage"

    if usage:
        # ✅ 개선: 소스에 따라 필드명 분기
        if source == "usage_metadata":
            prompt_tokens = usage.get('input_tokens', 0)
            completion_tokens = usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
        else:
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

        # Fallback: total_tokens이 없으면 계산
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        # 전체 누적
        _token_accumulator["prompt"] += prompt_tokens
        _token_accumulator["completion"] += completion_tokens
        _token_accumulator["total"] += total_tokens

        # 단계별 저장 (누적)
        if context_name not in _step_tokens:
            _step_tokens[context_name] = {"prompt": 0, "completion": 0, "total": 0}
        _step_tokens[context_name]["prompt"] += prompt_tokens
        _step_tokens[context_name]["completion"] += completion_tokens
        _step_tokens[context_name]["total"] += total_tokens

        print(f"📥 입력 토큰 (prompt):     {prompt_tokens:,} tokens")
        print(f"📤 출력 토큰 (completion): {completion_tokens:,} tokens")
        print(f"📊 총 토큰 (total):        {total_tokens:,} tokens")
        print(f"🔍 토큰 소스: {source}")
    else:
        print(f"⚠️  토큰 사용량 정보를 찾을 수 없습니다.")
        print(f"응답 객체 속성: {dir(response)}")
        if hasattr(response, 'response_metadata'):
            print(f"response_metadata: {response.response_metadata}")
        if hasattr(response, 'usage_metadata'):
            print(f"usage_metadata: {response.usage_metadata}")

    print(f"{'='*60}\n")

def print_recipe_token_brief():
    """레시피 생성 토큰 사용량 간단 요약 (🔷 박스)"""
    has_tokens = _token_accumulator["total"] > 0

    if not has_tokens:
        return

    print(f"\n{'🔷'*30}")
    print(f"{'  '*10}📊 레시피 생성 토큰 사용량 요약")
    print(f"{'🔷'*30}")
    print(f"📥 총 입력 토큰 (prompt):     {_token_accumulator['prompt']:,} tokens")
    print(f"📤 총 출력 토큰 (completion): {_token_accumulator['completion']:,} tokens")
    print(f"📊 총합 (total):              {_token_accumulator['total']:,} tokens")
    print(f"{'🔷'*30}\n")


def print_recipe_token_detail():
    """레시피 생성 토큰 사용량 상세 표 출력"""
    has_tokens = _token_accumulator["total"] > 0
    has_timings = len(_step_timings) > 0

    if not has_tokens and not has_timings:
        return

    # ✅ 1) 단계별 토큰 요약 표 (마크다운)
    if has_tokens:
        print("\n" + "="*100)
        print("- 📋 단계별 상세 요약\n")
        print("| Step | 설명 | Prompt Tokens | Completion Tokens | Total Tokens |")
        print("|------|------|---------------|-------------------|--------------|")

        # 단계 순서 정의
        step_order = ["검색 쿼리 추출", "레시피 생성"]
        step_metadata = {
            "검색 쿼리 추출": {"step": "1", "desc": "검색 쿼리 추출"},
            "레시피 생성": {"step": "2", "desc": "레시피 생성"},
        }

        # 단계 순서대로 출력
        for step_name in step_order:
            tokens = _step_tokens.get(step_name, {"prompt": 0, "completion": 0, "total": 0})
            meta = step_metadata.get(step_name, {"step": "-", "desc": step_name})

            if tokens["total"] > 0:
                prompt_str = str(tokens["prompt"]) if tokens["prompt"] > 0 else "-"
                completion_str = str(tokens["completion"]) if tokens["completion"] > 0 else "-"
                total_str = str(tokens["total"]) if tokens["total"] > 0 else "-"
                print(f"| {meta['step']} | {meta['desc']} | {prompt_str} | {completion_str} | {total_str} |")

        # ✅ 2) 전체 합계 요약 표 (마크다운)
        print("\n- 📊 전체 합계 요약\n")
        print("| 구분 | Prompt Tokens | Completion Tokens | Total Tokens |")
        print("|------|---------------|-------------------|--------------|")
        print(f"| 합계 | {_token_accumulator['prompt']:,} | {_token_accumulator['completion']:,} | {_token_accumulator['total']:,} |")

    # ✅ 3) 성능 병목 표: 동작 플로우 순서대로 (마크다운)
    if has_timings:
        print("\n- ⚡ 성능 병목 분석\n")
        print("| 동작 | 단계 | Latency(s) | 비율 |")
        print("|------|------|------------|------|")

        # 동작 순서 정의 (플로우 순서)
        step_order = ["검색 쿼리 추출", "레시피 생성"]
        total_time = sum(_step_timings.values())

        for order, step_name in enumerate(step_order, 1):
            ms = _step_timings.get(step_name, 0)
            if ms > 0:
                sec = ms / 1000
                ratio = (ms / total_time * 100) if total_time > 0 else 0
                print(f"| {order} | {step_name} | {sec:.1f} | ~{ratio:.0f}% |")

        # 총 소요 시간 추가
        total_sec = total_time / 1000
        print(f"| - | **TOTAL** | **{total_sec:.1f}** | **100%** |")

    print("="*100 + "\n")

    # 초기화
    _token_accumulator["prompt"] = 0
    _token_accumulator["completion"] = 0
    _token_accumulator["total"] = 0
    _step_tokens.clear()
    _step_timings.clear()


def print_recipe_token_summary():
    """레시피 생성 토큰 사용량 요약 출력 (하위 호환성 유지)"""
    print_recipe_token_brief()
    print_recipe_token_detail()


def _parse_recipe_response(response_text: str, servings: int = 1) -> dict:
    """LLM 응답을 TOON 우선 → JSON fallback으로 파싱"""
    # 마크다운 코드 블록 제거
    cleaned = re.sub(r'```(?:json|toon)?\s*|\s*```', '', response_text).strip()

    def _parse_toon_fallback(text: str) -> dict:
        """간단한 TOON 라인 파서 (toon_format 실패 시 fallback)"""
        # TOON: 접두 제거
        if text.startswith("TOON:"):
            text = text.split("TOON:", 1)[1].strip()
        lines = [ln.rstrip() for ln in text.splitlines()]

        data: Dict[str, Any] = {
            "title": "",
            "intro": "",
            "cook_time": "",
            "level": "",
            "servings": f"{servings}인분",
            "ingredients": [],
            "steps": [],
        }
        mode = None  # "ingredients" | "steps"

        def is_section_line(line: str) -> bool:
            return bool(re.match(r'^\s*\w+\s*:', line))

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            # 섹션 헤더
            if re.match(r'^ingredients\s*\[', line):
                mode = "ingredients"
                continue
            if re.match(r'^steps\s*\[', line):
                mode = "steps"
                continue

            # 키: 값 형태
            if is_section_line(line):
                mode = None
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key in data:
                    data[key] = val
                continue

            # 섹션 내용
            if mode == "ingredients":
                item = line.lstrip("-* ").strip()
                if not item:
                    continue
                parts = [p.strip() for p in item.split(",", 2)]
                if len(parts) >= 2:
                    name = parts[0]
                    amount = parts[1]
                    note = parts[2] if len(parts) >= 3 else ""
                    data["ingredients"].append(
                        {"name": name, "amount": amount, "note": note}
                    )
                continue

            if mode == "steps":
                item = line.lstrip("-* ").strip()
                if not item:
                    continue
                if "," in item:
                    no_str, desc = item.split(",", 1)
                    no_str = no_str.strip()
                    desc = desc.strip()
                else:
                    m = re.match(r'^(\d+)[\.\)]?\s*(.+)$', item)
                    if not m:
                        continue
                    no_str, desc = m.group(1), m.group(2)
                data["steps"].append({"no": no_str, "desc": desc})

        # 최소한의 유효성
        if not data["servings"]:
            data["servings"] = f"{servings}인분"
        return data

    # 1차: TOON 파싱 시도
    try:
        recipe = toon_decode(cleaned)
        if recipe.get('title') and recipe.get('ingredients'):
            print(f"[RecipeService] TOON 파싱 성공")
            return recipe
    except Exception as e:
        print(f"[RecipeService] TOON 파싱 실패: {e}")

    # 1.5차: TOON 간이 파싱 (fallback)
    try:
        recipe = _parse_toon_fallback(cleaned)
        if recipe.get('title') or recipe.get('ingredients') or recipe.get('steps'):
            print(f"[RecipeService] TOON fallback 파싱 성공")
            return recipe
    except Exception as e:
        print(f"[RecipeService] TOON fallback 파싱 실패: {e}")

    # 2차: JSON 파싱 시도 (LLM이 JSON으로 응답한 경우)
    try:
        recipe = json.loads(cleaned)
        if isinstance(recipe, dict):
            print(f"[RecipeService] JSON fallback 파싱 성공")
            return recipe
    except json.JSONDecodeError as e:
        print(f"[RecipeService] JSON 파싱도 실패: {e}")
        print(f"[RecipeService] 응답: {cleaned[:200]}")

    # 3차: 빈 레시피 반환
    return {
        "title": "추천 레시피",
        "intro": "레시피 생성 중 오류가 발생했습니다.",
        "cook_time": "30분",
        "level": "중급",
        "servings": f"{servings}인분",
        "ingredients": [],
        "steps": [],
    }


class RecipeService:
    def __init__(self, rag_system, recipe_db, user_profile=None):
        self.rag = rag_system
        self.db = recipe_db
        self.user_profile = user_profile or {}
    
    async def generate_recipe(
        self, 
        chat_history: List[Dict[str, str]],
        member_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """상세 레시피 생성 (대화 기반) + 이미지 URL"""
        
        print(f"[RecipeService] 레시피 생성 시작")
        print(f"[RecipeService] 대화 개수: {len(chat_history)}")
        print(f"[RecipeService] 가족 정보: {member_info}")
        
        # 1. LLM으로 대화 분석 + 검색 쿼리 생성
        search_query = self._extract_search_query_with_llm(chat_history, member_info)
        
        print(f"[RecipeService] 생성된 검색 쿼리: {search_query}")
        
        # 2. RAG 검색 (Neo4j - allergies/tools 필터 포함)
        allergies = member_info.get('allergies', []) if member_info else []
        user_tools = member_info.get('tools', []) if member_info else []
        retrieved_docs = self.rag.search_recipes(
            search_query, k=3, use_rerank=False,
            allergies=allergies, user_tools=user_tools
        )
        
        print(f"[RecipeService] RAG 검색 결과: {len(retrieved_docs)}개")
        
        # 웹 검색 여부 판단
        from_web_search = not retrieved_docs or len(retrieved_docs) == 0
        
        # 3. 알레르기/비선호 필터링
        filtered_docs = self._filter_by_constraints(retrieved_docs, member_info)
        
        print(f"[RecipeService] 필터링 후: {len(filtered_docs)}개")
        
        # 4. LLM으로 최종 레시피 생성
        recipe_json = self._generate_final_recipe_with_llm(
            chat_history=chat_history,
            member_info=member_info,
            context_docs=filtered_docs
        )
        
        print(f"[RecipeService] 레시피 생성 완료: {recipe_json.get('title')}")
        
        # 5. 이미지 찾기
        recipe_title = recipe_json.get('title', '')
        best_image = ""
        
        if from_web_search:
            # 웹 검색이면 기본 이미지
            print(f"[RecipeService] 웹 검색 레시피 → 기본 이미지 사용")
            best_image = '/default-food.jpg'
        else:
            # Neo4j 검색 결과 메타데이터에서 이미지 가져오기
            best_image = self._get_best_image(filtered_docs)
            if not best_image:
                print(f"[RecipeService] 이미지 없음 → 기본 이미지 사용")
                best_image = '/default-food.jpg'
        
        print(f"[RecipeService] 선택된 이미지: {best_image or '기본 이미지'}")
        
        # 6. 이미지 URL 추가
        recipe_json['image'] = best_image
        recipe_json['img_url'] = best_image
        
        # 7. 인원수 설정
        servings = len(member_info.get('names', [])) if member_info and member_info.get('names') else 1
        if 'servings' not in recipe_json or not recipe_json['servings']:
            recipe_json['servings'] = f"{servings}인분"
        
        print(f"[RecipeService] 최종 레시피: {recipe_json.get('title')}")
        print(f"[RecipeService] 인원수: {recipe_json['servings']}")
        print(f"[RecipeService] 이미지: {recipe_json.get('image', 'None')[:60]}...")
        
        return recipe_json

    async def generate_recipe_from_existing(
        self,
        recipe_content: str,
        member_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """기존 레시피에서 상세 조리 과정 생성 (RAG 없이)"""

        print(f"[RecipeService] 기존 레시피로부터 상세 레시피 생성 시작")
        print(f"[RecipeService] 레시피 내용: {recipe_content[:200]}...")

        # 1. LLM으로 상세 레시피 생성 (title 포함)
        recipe_json = self._expand_recipe_with_llm(
            recipe_content=recipe_content,
            member_info=member_info
        )

        print(f"[RecipeService] 상세 레시피 생성 완료: {recipe_json.get('title')}")

        # 2. 기본 이미지 사용 (기존 레시피 확장은 RAG 검색 없음)
        best_image = '/default-food.jpg'

        print(f"[RecipeService] 선택된 이미지: {best_image[:60]}...")

        # 4. 이미지 URL 추가
        recipe_json['image'] = best_image
        recipe_json['img_url'] = best_image

        # 5. 인원수 설정
        servings = len(member_info.get('names', [])) if member_info and member_info.get('names') else 1
        if 'servings' not in recipe_json or not recipe_json['servings']:
            recipe_json['servings'] = f"{servings}인분"

        print(f"[RecipeService] 최종 레시피: {recipe_json.get('title')}")
        print(f"[RecipeService] 인원수: {recipe_json['servings']}")
        print(f"[RecipeService] 이미지: {recipe_json.get('image', 'None')[:60]}...")

        return recipe_json

    def _extract_title_from_recipe(self, recipe_content: str) -> str:
        """레시피 내용에서 제목 추출"""
        import re

        # 1. **[요리명]** 형식 찾기
        match = re.search(r'\*\*\[([^\]]+)\]\*\*', recipe_content)
        if match:
            return match.group(1).strip()

        # 2. 줄의 시작에서 [요리명] 형식 찾기 (재료 리스트 안의 대괄호 제외)
        # 줄 시작 또는 줄바꿈 후에만 매칭
        lines = recipe_content.split('\n')
        for line in lines[:5]:  # 처음 5줄만 확인
            line = line.strip()
            # 줄이 [로 시작하고 ]로 끝나는 경우
            match = re.match(r'^\[([^\]]+)\]$', line)
            if match:
                title = match.group(1).strip()
                # 섹션 헤더나 수정 메모는 제외
                exclude_patterns = ['소개:', '재료:', '조리법:', '팁:', '빼고', '변경', '추가', '제외', '제거', '대신', '말고']
                if not any(pattern in title for pattern in exclude_patterns):
                    return title

        # 3. **요리명** 형식 찾기 (섹션 헤더 제외)
        matches = re.findall(r'\*\*([^*\n]+)\*\*', recipe_content)
        for title in matches:
            title = title.strip()
            # 섹션 헤더, 이모지, 숫자로 시작하는 것 제외
            if not re.match(r'^[⏱️📊👥\d]', title) and ':' not in title:
                return title

        # 4. 첫 번째 줄을 제목으로 간주
        first_line = recipe_content.split('\n')[0].strip()
        # 특수 문자 제거
        first_line = re.sub(r'[*\[\]#]', '', first_line).strip()
        return first_line

    def _expand_recipe_with_llm(
        self,
        recipe_content: str,
        member_info: Dict
    ) -> Dict:
        """LLM으로 기존 레시피를 상세 조리 과정으로 확장"""

        servings = len(member_info.get('names', [])) if member_info else 1
        tools = ', '.join(member_info.get('tools', [])) if member_info else '모든 도구'

        # 프롬프트 사용
        prompt = RECIPE_DETAIL_EXPANSION_PROMPT.format(
            recipe_content=recipe_content,
            servings=servings,
            tools=tools
        )

        from langchain_naver import ChatClovaX
        llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=2000)

        try:
            result = llm.invoke(prompt)
            response_text = result.content.strip()

            # TOON 우선 → JSON fallback 파싱
            recipe_json = _parse_recipe_response(response_text, servings)

            print(f"[RecipeService] 상세 레시피 생성 성공: {recipe_json.get('title')}")
            return recipe_json

        except Exception as e:
            print(f"[RecipeService] 상세 레시피 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _get_best_image(self, filtered_docs: List[Dict]) -> str:
        """Neo4j 검색 결과 메타데이터에서 이미지 URL 가져오기"""
        for doc in filtered_docs:
            img = (doc.get("metadata") or {}).get("image_url", "")
            if img:
                print(f"[RecipeService] Neo4j 이미지 URL: {img[:60]}...")
                return img
        return ""
    
    def _extract_search_query_with_llm(
        self, 
        chat_history: List[Dict],
        member_info: Dict
    ) -> str:
        """LLM으로 검색 쿼리 추출"""
        
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history[-10:]
        ])
        
        servings = len(member_info.get('names', [])) if member_info else 1
        allergies = ', '.join(member_info.get('allergies', [])) if member_info else '없음'
        dislikes = ', '.join(member_info.get('dislikes', [])) if member_info else '없음'
        
        # 프롬프트 사용
        prompt = RECIPE_QUERY_EXTRACTION_PROMPT.format(
            conversation=conversation,
            servings=servings,
            allergies=allergies,
            dislikes=dislikes
        )
        
        from langchain_naver import ChatClovaX
        llm = ChatClovaX(model="HCX-DASH-001", temperature=0.2, max_tokens=50)
        
        try:
            result = llm.invoke(prompt)
            print_token_usage(result, "검색 쿼리 추출")

            query = result.content.strip()
            print(f"[RecipeService] LLM 추출 쿼리: {query}")
            return query
        except Exception as e:
            print(f"[RecipeService] 쿼리 추출 실패: {e}")
            return self._simple_keyword_extraction(chat_history)
    
    def _simple_keyword_extraction(self, chat_history: List[Dict]) -> str:
        """간단한 키워드 추출 (Fallback)"""
        food_keywords = []
        
        for msg in chat_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                if any(k in content for k in ['찌개', '국', '탕', '볶음', '구이', '조림']):
                    words = content.split()
                    food_keywords.extend([w for w in words if len(w) > 1])
        
        return ' '.join(food_keywords[:5]) if food_keywords else "한식 요리"
    
    def _filter_by_constraints(
        self,
        recipes: List[Dict],
        member_info: Dict
    ) -> List[Dict]:
        """알레르기/비선호 필터링"""
        
        if not member_info:
            return recipes[:5]
        
        filtered = []
        
        for recipe in recipes:
            content = recipe.get("content", "").lower()
            
            # 알레르기 체크
            if member_info.get("allergies"):
                has_allergen = any(
                    allergen.lower() in content 
                    for allergen in member_info["allergies"]
                )
                if has_allergen:
                    continue
            
            # 비선호 재료 체크
            if member_info.get("dislikes"):
                has_dislike = any(
                    dislike.lower() in content 
                    for dislike in member_info["dislikes"]
                )
                if has_dislike:
                    continue
            
            filtered.append(recipe)
            
            if len(filtered) >= 5:
                break
        
        if len(filtered) < 3:
            return recipes[:3]
        
        return filtered
    
    def _generate_final_recipe_with_llm(
        self,
        chat_history: List[Dict],
        member_info: Dict,
        context_docs: List[Dict]
    ) -> Dict:
        """LLM으로 최종 레시피 JSON 생성"""
        
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history
        ])
        
        context_text = "\n\n".join([
            f"[레시피 {i+1}] {doc.get('title')}\n{doc.get('content', '')[:800]}"
            for i, doc in enumerate(context_docs[:5])
        ])
        
        servings = len(member_info.get('names', [])) if member_info else 1
        allergies = ', '.join(member_info.get('allergies', [])) if member_info else '없음'
        dislikes = ', '.join(member_info.get('dislikes', [])) if member_info else '없음'
        tools = ', '.join(member_info.get('tools', [])) if member_info else '모든 도구'
        
        # 프롬프트 사용
        prompt = RECIPE_GENERATION_PROMPT.format(
            conversation=conversation,
            servings=servings,
            allergies=allergies,
            dislikes=dislikes,
            tools=tools,
            context=context_text
        )
        
        from langchain_naver import ChatClovaX
        llm = ChatClovaX(model="HCX-DASH-003", temperature=0.2, max_tokens=2000)
        
        try:
            result = llm.invoke(prompt)
            print_token_usage(result, "레시피 생성")

            response_text = result.content.strip()

            # TOON 우선 → JSON fallback 파싱
            recipe_json = _parse_recipe_response(response_text, servings)

            print(f"[RecipeService] 레시피 생성 성공: {recipe_json.get('title')}")
            return recipe_json

        except Exception as e:
            print(f"[RecipeService] 레시피 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise