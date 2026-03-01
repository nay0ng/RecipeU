# backend/features/chat/agent.py
"""
Chat Agent - Adaptive RAG
"""
import os
import time
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# prompts.py에서 프롬프트 import
from .prompts import REWRITE_PROMPT, GRADE_PROMPT, GENERATE_PROMPT
from services.search import get_search_service


# ─────────────────────────────────────────────
# 토큰 사용량 추적 헬퍼 함수
# ─────────────────────────────────────────────
# 요청별 토큰 누적 (요청당 초기화됨)
_token_accumulator: dict = {"prompt": 0, "completion": 0, "total": 0}
# 노드별 토큰 정보 저장 (노드명 -> {prompt, completion, total})
_node_tokens: dict = {}

def print_token_usage(response, context_name: str = "LLM"):
    """LLM 응답에서 실제 토큰 사용량 출력 및 누적 (개선 버전)"""
    print(f"\n{'='*60}")
    print(f"[{context_name}] HCX API 토큰 사용량 (실측)")
    print(f"{'='*60}")

    # 개선: usage_metadata 우선 확인 (LangChain 표준)
    usage = None
    source = ""

    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        source = "usage_metadata"
    elif hasattr(response, 'response_metadata'):
        usage = response.response_metadata.get('token_usage')
        source = "response_metadata.token_usage"

    if usage:
        # 개선: 소스에 따라 필드명 분기
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

        # 노드별 저장 (누적)
        if context_name not in _node_tokens:
            _node_tokens[context_name] = {"prompt": 0, "completion": 0, "total": 0}
        _node_tokens[context_name]["prompt"] += prompt_tokens
        _node_tokens[context_name]["completion"] += completion_tokens
        _node_tokens[context_name]["total"] += total_tokens

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

def print_token_summary():
    """누적된 토큰 사용량 요약 출력 (마크다운 형식)"""
    if _token_accumulator["total"] == 0:
        return

    print(f"\n{'🔷'*30}")
    print(f"{'  '*10}📊 전체 토큰 사용량 요약")
    print(f"{'🔷'*30}")
    print(f"📥 총 입력 토큰 (prompt):     {_token_accumulator['prompt']:,} tokens")
    print(f"📤 총 출력 토큰 (completion): {_token_accumulator['completion']:,} tokens")
    print(f"📊 총합 (total):              {_token_accumulator['total']:,} tokens")
    print(f"{'🔷'*30}\n")

    # 1) 노드별 토큰/시간 요약 표 (마크다운)
    print("\n" + "="*100)
    print("- 📋 노드별 상세 요약\n")
    print("| Step | Node | 설명 | Prompt Tokens | Completion Tokens | Total Tokens | Latency(s) |")
    print("|------|------|------|---------------|-------------------|--------------|------------|")

    # 노드 순서 및 메타데이터 정의
    node_order = ["관련성 체크", "쿼리 재작성", "retrieve", "check_constraints", "관련성 평가", "web_search", "제약 조건 경고", "답변 생성"]
    node_metadata = {
        "관련성 체크": {"step": "0", "desc": "레시피 관련성 체크", "timing_key": "check_relevance"},
        "쿼리 재작성": {"step": "1", "desc": "쿼리 재작성", "timing_key": "rewrite"},
        "retrieve": {"step": "2", "desc": "RAG 검색", "timing_key": "retrieve"},
        "check_constraints": {"step": "2.5", "desc": "제약 조건 체크", "timing_key": "check_constraints"},
        "관련성 평가": {"step": "3", "desc": "문서 관련성 평가", "timing_key": "grade"},
        "web_search": {"step": "4", "desc": "웹 검색", "timing_key": "web_search"},
        "제약 조건 경고": {"step": "5a", "desc": "제약 조건 경고", "timing_key": "generate"},
        "답변 생성": {"step": "5", "desc": "답변 생성", "timing_key": "generate"},
    }

    # 노드 순서대로 출력
    for node_name in node_order:
        tokens = _node_tokens.get(node_name, {"prompt": 0, "completion": 0, "total": 0})
        meta = node_metadata.get(node_name, {"step": "-", "desc": node_name, "timing_key": node_name})
        timing_ms = _node_timings.get(meta["timing_key"], 0)
        timing_sec = timing_ms / 1000 if timing_ms else 0

        if tokens["total"] > 0 or timing_sec > 0:
            prompt_str = str(tokens["prompt"]) if tokens["prompt"] > 0 else "-"
            completion_str = str(tokens["completion"]) if tokens["completion"] > 0 else "-"
            total_str = str(tokens["total"]) if tokens["total"] > 0 else "-"
            latency_str = f"{timing_sec:.1f}" if timing_sec > 0 else "-"
            print(f"| {meta['step']} | {node_name} | {meta['desc']} | {prompt_str} | {completion_str} | {total_str} | {latency_str} |")

    # 2) 전체 합계 요약 표 (마크다운)
    print("\n- 📊 전체 합계 요약\n")
    print("| 구분 | Prompt Tokens | Completion Tokens | Total Tokens |")
    print("|------|---------------|-------------------|--------------|")
    print(f"| 합계 | {_token_accumulator['prompt']:,} | {_token_accumulator['completion']:,} | {_token_accumulator['total']:,} |")

    # 3) 성능 병목 표: 동작 플로우 순서대로 (마크다운)
    if _node_timings:
        print("\n- ⚡ 성능 병목 분석\n")
        print("| 동작 | Node | Latency(s) | 비율 |")
        print("|------|------|------------|------|")

        # 동작 플로우 순서 정의
        node_order = ["check_relevance", "rewrite", "retrieve", "check_constraints", "grade", "web_search", "generate"]
        total_time = sum(_node_timings.values())

        for order, node_name in enumerate(node_order, 1):
            ms = _node_timings.get(node_name, 0)
            if ms > 0:
                sec = ms / 1000
                ratio = (ms / total_time * 100) if total_time > 0 else 0
                print(f"| {order} | {node_name} | {sec:.1f} | ~{ratio:.0f}% |")

        # 총 소요 시간 추가
        total_sec = total_time / 1000
        print(f"| - | **TOTAL** | **{total_sec:.1f}** | **100%** |")

        print("="*100 + "\n")

    # 초기화
    _token_accumulator["prompt"] = 0
    _token_accumulator["completion"] = 0
    _token_accumulator["total"] = 0
    _node_tokens.clear()
    _node_timings.clear()


# ─────────────────────────────────────────────
# 노드별 타이밍 래퍼
# ─────────────────────────────────────────────
# 누적 타이밍을 저장할 전역 딕셔너리 (요청당 초기화됨)
_node_timings: dict = {}

def timed_node(name: str, fn):
    """노드 함수를 감싸서 실행 시간을 자동 로깅"""
    def wrapper(state: "ChatAgentState") -> "ChatAgentState":
        start = time.time()
        result = fn(state)
        elapsed_ms = (time.time() - start) * 1000
        _node_timings[name] = elapsed_ms
        elapsed_sec = elapsed_ms / 1000
        print(f"  ⏱️  [Node: {name}] {elapsed_sec:.1f}초")
        return result
    return wrapper


class ChatAgentState(TypedDict):
    """Agent 상태"""
    question: str
    original_question: str
    chat_history: List[str]
    documents: List[Document]
    generation: str
    web_search_needed: str
    user_constraints: dict
    constraint_warning: str
    modification_history: list  # 레시피 수정 이력


def create_chat_agent(rag_system):
    """Chat Agent 생성 - Adaptive RAG + 네이버 검색"""

    search_engine = os.getenv("SEARCH_ENGINE", "serper")
    search_service = get_search_service(search_engine)
    print(f"[Agent] 검색 엔진: {search_engine}")
    
    # ===== 노드 함수 =====

    def rewrite_query(state: ChatAgentState) -> ChatAgentState:
        """쿼리 재작성"""
        print("[Agent] 쿼리 재작성 중...")

        question = state["question"]
        history = state.get("chat_history", [])
        user_constraints = state.get("user_constraints", {})

        formatted_history = "\n".join(history[-5:]) if isinstance(history, list) else str(history)

        try:
            from langchain_naver import ChatClovaX
            llm = ChatClovaX(model="HCX-DASH-001", temperature=0.2, max_tokens=50)
            chain = REWRITE_PROMPT | llm
            _rewrite_response = chain.invoke({
                "history": formatted_history,
                "question": question
            })
            print_token_usage(_rewrite_response, "쿼리 재작성")
            better_question = _rewrite_response.content.strip()

            print(f"   원본: {question}")
            print(f"   재작성: {better_question}")

            # 재작성 결과가 원본보다 3배 이상 길거나 문장형이면 원본 사용
            if len(better_question) > len(question) * 3 or any(kw in better_question for kw in ["확인되지", "않습니다", "말씀하신", "궁금하신"]):
                print(f"   재작성 결과 이상 → 원본 사용")
                better_question = question

            # 알레르기/비선호 키워드를 재작성된 쿼리에서 제거
            # (grade_documents와 web_search가 필터링된 쿼리를 기준으로 동작하게 함)
            allergies = user_constraints.get("allergies", []) or []
            dislikes = user_constraints.get("dislikes", []) or []
            excluded_terms = allergies + dislikes

            if excluded_terms:
                filtered_question = better_question
                for term in excluded_terms:
                    filtered_question = filtered_question.replace(term, "").strip()
                # 필터링 후 공백 정리
                import re as _re
                filtered_question = _re.sub(r'\s+', ' ', filtered_question).strip()
                if filtered_question:
                    if filtered_question != better_question:
                        print(f"   알레르기/비선호 제거: '{better_question}' → '{filtered_question}'")
                    better_question = filtered_question

            return {
                "question": better_question,
                "original_question": question
            }

        except Exception as e:
            print(f"   재작성 실패: {e}")
            return {
                "question": question,
                "original_question": question
            }

    def retrieve(state: ChatAgentState) -> ChatAgentState:
        """RAG 검색 (Reranker 사용)"""
        print("[Agent] RAG 검색 중...")

        question = state["question"]
        user_constraints = state.get("user_constraints", {})
        allergies = user_constraints.get("allergies", []) or None
        user_tools = user_constraints.get("cooking_tools", []) or None

        if user_tools:
            print(f"  🔧 조리도구 필터: {user_tools}")

        # use_rerank=None -> RAG 시스템 설정(USE_RERANKER) 따름
        results = rag_system.search_recipes(question, k=3, use_rerank=None, allergies=allergies, user_tools=user_tools)
        
        documents = [
            Document(
                page_content=doc.get("content", ""),
                metadata={
                    "title": doc.get("title", ""),
                    "cook_time": doc.get("cook_time", ""),
                    "level": doc.get("level", ""),
                    "cooking_tools": doc.get("cooking_tools", []),
                }
            )
            for doc in results
        ]
        
        print(f"   검색 결과: {len(documents)}개")
        for i, doc in enumerate(documents[:3], 1):
            print(f"   {i}. {doc.metadata.get('title', '')[:40]}...")
        
        return {"documents": documents}
    
    def check_constraints(state: ChatAgentState) -> ChatAgentState:
        """제약 조건 체크 (알레르기, 비선호 음식)"""
        print("[Agent] 제약 조건 체크 중...")

        # original_question 기준으로 확인: rewrite 후 필터링된 question은 이미 알레르기 제거됨
        # 사용자가 알레르기 재료를 명시적으로 요청했는지는 원본 질문으로 판단해야 함
        original_question = state.get("original_question") or state["question"]
        user_constraints = state.get("user_constraints", {})

        if not user_constraints:
            print("   제약 조건 없음 → 스킵")
            return {"constraint_warning": ""}

        dislikes = user_constraints.get("dislikes", [])
        allergies = user_constraints.get("allergies", [])

        question_lower = original_question.lower()
        warning_parts = []

        import re
        for allergy in allergies:
            # 단어 단위 매칭: "게" → "게"만 매칭, "맵게"는 매칭 안 됨
            pattern = r'(?<![가-힣])' + re.escape(allergy.lower()) + r'(?![가-힣])'
            if re.search(pattern, question_lower):
                warning_parts.append(f"**{allergy}**는 알레르기 재료입니다!")

        for dislike in dislikes:
            pattern = r'(?<![가-힣])' + re.escape(dislike.lower()) + r'(?![가-힣])'
            if re.search(pattern, question_lower):
                warning_parts.append(f"**{dislike}**는 싫어하는 음식입니다.")
    
        if warning_parts:
            warning_msg = "\n".join(warning_parts)
            print(f"   제약 조건 위반 감지!")
            print(f"   {warning_msg}")
            return {"constraint_warning": warning_msg}
        else:
            print("   제약 조건 통과")
            return {"constraint_warning": ""}

    def grade_documents(state: ChatAgentState) -> ChatAgentState:
        """문서 관련성 평가"""
        print("[Agent] 관련성 평가 중...")
        
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            print("   문서 없음 → 웹 검색")
            return {"web_search_needed": "yes"}
        
        try:
            question_lower = question.lower()

            found_exact_match = False
            for doc in documents[:3]:
                title = doc.metadata.get("title", "").lower()
                if question_lower in title or any(
                    word in title
                    for word in question_lower.split()
                    if len(word) > 1
                ):
                    found_exact_match = True
                    break
            
            if not found_exact_match:
                print("   제목 매칭 실패 → 웹 검색")
                return {"web_search_needed": "yes"}
            
            context_text = "\n".join([
                f"- {doc.page_content[:200]}"
                for doc in documents[:3]
            ])

            from langchain_naver import ChatClovaX
            llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=50)
            chain = GRADE_PROMPT | llm
            _grade_response = chain.invoke({
                "question": question,
                "context": context_text
            })
            print_token_usage(_grade_response, "관련성 평가")
            score = _grade_response.content.strip()

            print(f"   평가: {score}")

            # HCX-003이 max_tokens 제한 무시하고 긴 응답 반환하는 경우 대비:
            # "no" 명시적 포함 시만 웹검색, 나머지(긴 한국어 포함)는 DB 충분으로 처리
            # (이미 제목 키워드 매칭 통과한 상태이므로 기본값은 DB 충분)
            if "no" in score.lower():
                print("   DB 부족 → 웹 검색")
                return {"web_search_needed": "yes"}
            else:
                print("   DB 충분 → 생성")
                return {"web_search_needed": "no"}
                
        except Exception as e:
            print(f"   평가 실패: {e}")
            return {"web_search_needed": "yes"}
    
    def web_search(state: ChatAgentState) -> ChatAgentState:
        """웹 검색"""
        print("[Agent] 웹 검색 실행 중...")

        question = state["question"]
        documents = search_service.search(query=question, max_results=3)

        for i, doc in enumerate(documents, 1):
            print(f"\n   [검색 결과 {i}]")
            print(f"   제목: {doc.metadata.get('title', '')}")
            print(f"   내용: {doc.page_content[:200]}...")

        return {"documents": documents}

    def summarize_web_results(state: ChatAgentState) -> ChatAgentState:
        """웹 검색 결과 요약"""
        print("[Agent] 웹 검색 결과 요약 중...")

        question = state["question"]
        documents = state["documents"]

        if not documents:
            print("   요약할 문서 없음")
            return {"documents": documents}

        try:
            summarized_docs = []

            for i, doc in enumerate(documents, 1):
                # 각 문서를 간결하게 요약
                summarize_prompt = f"""질문: {question}

내용:
{doc.page_content[:800]}

**요약 (3문장, 재료/시간/난이도 위주, 광고 제거, 정확한 양 유지):**"""

                from langchain_naver import ChatClovaX
                from langchain_core.messages import HumanMessage
                llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=300)
                result = llm.invoke([HumanMessage(content=summarize_prompt)])
                summary = result.content.strip()

                summarized_doc = Document(
                    page_content=summary,
                    metadata=doc.metadata
                )
                summarized_docs.append(summarized_doc)

                print(f"   {i}. 요약 완료: {summary[:50]}...")

            return {"documents": summarized_docs}

        except Exception as e:
            print(f"   요약 실패: {e}, 원본 사용")
            return {"documents": documents}

    def generate(state: ChatAgentState) -> ChatAgentState:
        """답변 생성"""
        print("[Agent] 답변 생성 중...")

        # 검색/grade에 사용된 필터링 쿼리를 사용 (original_question 대신)
        # original_question("새우볶음밥")은 컨텍스트("햄볶음밥" 등)와 불일치하여
        # LLM이 컨텍스트를 무시하고 임의 레시피를 생성하는 원인이 됨
        question = state.get("question") or state["original_question"]
        documents = state["documents"]
        history = state.get("chat_history", [])
        constraint_warning = state.get("constraint_warning", "")
        user_constraints = state.get("user_constraints", {})

        formatted_history = "\n".join(history[-10:]) if isinstance(history, list) else str(history)

        # 컨텍스트 구성: TOON 형식으로 입력 토큰 절약
        # title·cooking_tools는 TOON 헤더로 표현하고 page_content에서 중복 제거
        context_parts = []
        for doc in documents:
            title = doc.metadata.get("title", "")
            cook_time = doc.metadata.get("cook_time", "")
            level = doc.metadata.get("level", "")
            tools = doc.metadata.get("cooking_tools", [])

            toon_lines = []
            if title:
                toon_lines.append(f"title: {title}")
            if cook_time and cook_time not in ("", "N/A"):
                toon_lines.append(f"cook_time: {cook_time}")
            if level and level not in ("", "N/A"):
                toon_lines.append(f"level: {level}")
            if tools:
                toon_lines.append(f"cooking_tools: {','.join(tools)}")

            # page_content에서 title·조리도구 중복 줄 제거 후 500자 압축
            content_lines = doc.page_content.split('\n')
            if content_lines and content_lines[0].strip() == title:
                content_lines = content_lines[1:]
            content_lines = [l for l in content_lines if not l.startswith('조리도구:')]
            content = '\n'.join(content_lines).strip()
            if len(content) > 500:
                content = content[:500]
            if content:
                toon_lines.append(content)

            context_parts.append("\n".join(toon_lines))
        context_text = "\n---\n".join(context_parts)

        # constraint_warning이 있어도 생성을 차단하지 않음
        # Neo4j가 이미 DB 레벨에서 알레르기 재료를 필터링했으므로 계속 진행
        # (경고 내용은 enhanced_question에 포함되어 LLM에게 전달됨)

        try:
            # 제약 조건을 질문에 통합 (컨텍스트가 아닌 질문에 포함)
            enhanced_question = question
            available_tools_str = "제한 없음"
            if user_constraints:
                allergies = user_constraints.get("allergies", [])
                dislikes = user_constraints.get("dislikes", [])
                cooking_tools = user_constraints.get("cooking_tools", [])

                constraints = []
                if allergies:
                    constraints.append(f"제외: {', '.join(allergies)}")
                if dislikes:
                    constraints.append(f"비선호: {', '.join(dislikes)}")

                if constraints:
                    enhanced_question = f"{question} ({' / '.join(constraints)})"

                if cooking_tools:
                    available_tools_str = ', '.join(cooking_tools)

            # 인원수 계산 (선택한 가족 구성원 수)
            servings = 1  # 기본값
            if user_constraints:
                names = user_constraints.get("names", [])
                if names and len(names) > 0:
                    servings = len(names)

            print(f"   [인원수] {servings}인분으로 레시피 생성")

            # 수정 이력 처리 (재생성 시 이전 수정사항 반영)
            # "빼달라"거나 "없는" 재료만 반영 (추가 요청은 제외)
            modification_history = state.get("modification_history", [])
            modification_constraints = ""

            print(f"\n{'='*60}")
            print(f"[수정 이력 확인] 전체 수정 이력: {len(modification_history)}개")
            if modification_history:
                for i, mod in enumerate(modification_history, 1):
                    print(f"  [{i}] type={mod.get('type')}, request='{mod.get('request')}', remove={mod.get('remove_ingredients', [])}, add={mod.get('add_ingredients', [])}")

            if modification_history and len(modification_history) > 0:
                constrained_ingredients = []
                allowed_ingredients = []  # replace 타입에서 추가된 재료 (제약 해제)
                filtered_out = []

                for mod in modification_history:
                    mod_type = mod.get("type")

                    # remove(빼기) 또는 replace(대체)만 반영
                    if mod_type in ["remove", "replace"]:
                        remove_items = mod.get("remove_ingredients", [])
                        add_items = mod.get("add_ingredients", [])

                        if remove_items:
                            # 제거할 재료를 제약사항에 추가
                            constrained_ingredients.extend(remove_items)
                            print(f"  제약 추가: {remove_items} (type={mod_type})")

                        if add_items and mod_type == "replace":
                            # replace의 경우 추가된 재료는 제약 해제
                            allowed_ingredients.extend(add_items)
                            print(f"  제약 해제: {add_items} (type={mod_type}, 이제 사용 가능)")

                        if not remove_items and not add_items:
                            # 재료 추출 실패 시 제약사항에 추가하지 않음
                            print(f"  재료 추출 실패로 제약사항 추가 스킵: '{mod['request']}' (type={mod_type})")
                    else:
                        filtered_out.append(mod['request'])
                        print(f"  제외: {mod['request']} (type={mod_type})")

                # 제약 해제된 재료는 제약사항에서 제거
                if allowed_ingredients:
                    print(f"\n[제약 해제] {allowed_ingredients} → 제약사항에서 제거")
                    constrained_ingredients = [
                        ing for ing in constrained_ingredients
                        if ing not in allowed_ingredients
                    ]

                # 중복 제거
                constrained_ingredients = list(set(constrained_ingredients))

                if constrained_ingredients:
                    # 재료명 리스트로 제약사항 문구 생성
                    ingredients_text = ", ".join(constrained_ingredients)
                    modification_constraints = f"\n**이전 수정사항 (반드시 반영):**\n- 제외: {ingredients_text}\n"
                    print(f"\n[최종 제약사항] {len(constrained_ingredients)}개 재료 반영됨: {ingredients_text}")
                else:
                    print(f"\n[최종 제약사항] 반영할 제약사항 없음 (모두 필터링됨)")

                if filtered_out:
                    print(f"[제외된 수정사항] {len(filtered_out)}개: {', '.join(filtered_out)}")
            else:
                print(f"[최종 제약사항] 수정 이력 없음")
            print(f"{'='*60}\n")

            # TOON 출력은 마크다운보다 compact → max_tokens 절감
            from langchain_naver import ChatClovaX
            llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=400)
            chain = GENERATE_PROMPT | llm
            _generate_response = chain.invoke({
                "context": context_text,
                "question": enhanced_question,
                "history": formatted_history,
                "servings": servings,
                "modification_constraints": modification_constraints,
                "available_tools": available_tools_str,
            })
            print_token_usage(_generate_response, "답변 생성")
            answer = _generate_response.content.strip()
            print(f"\n[DEBUG] LLM 원본 응답:\n{answer}\n[/DEBUG]\n")

            import re

            def _parse_toon(text: str) -> dict:
                """TOON 텍스트 → dict. 두 가지 ingredients 포맷 지원:
                  - 다중 줄: 재료명,양 (한 줄에 하나)
                  - 단일 줄: 재료명 양, 재료명 양, ... (모두 한 줄)
                """
                data: dict = {
                    "title": "", "cook_time": "", "level": "",
                    "servings": "", "intro": "", "ingredients": [],
                    "cooking_tools": "",
                }
                mode = None

                for raw in text.splitlines():
                    line = raw.strip()
                    if not line:
                        continue

                    # ingredients 섹션 헤더 (ingredients[N]{...}: 또는 ingredients{...}: 모두 허용)
                    if re.match(r'^ingredients[\s\[\{:]', line, re.IGNORECASE):
                        mode = "ingredients"
                        inline = line.split(":", 1)[1].strip() if ":" in line else ""
                        if inline:
                            # 단일 줄 포맷: "재료명 양, 재료명 양, ..."
                            for token in re.split(r",\s*", inline):
                                parts = token.strip().split(None, 1)
                                if parts:
                                    data["ingredients"].append({
                                        "name": parts[0],
                                        "amount": parts[1] if len(parts) > 1 else "",
                                        "note": "",
                                    })
                        continue

                    # key: value 형태 (섹션 헤더)
                    if re.match(r'^\w+\s*:', line):
                        mode = None
                        key, val = line.split(":", 1)
                        key = key.strip()
                        val = val.strip()
                        if key in data:
                            data[key] = val
                        continue

                    # ingredients 다중 줄 포맷: "재료명,양" 또는 "재료명 양"
                    if mode == "ingredients":
                        item = line.lstrip("-* ").strip()
                        if not item:
                            continue
                        if "," in item:
                            parts = [p.strip() for p in item.split(",", 2)]
                            data["ingredients"].append({
                                "name": parts[0],
                                "amount": parts[1] if len(parts) > 1 else "",
                                "note": parts[2] if len(parts) > 2 else "",
                            })
                        else:
                            parts = item.split(None, 1)
                            data["ingredients"].append({
                                "name": parts[0],
                                "amount": parts[1] if len(parts) > 1 else "",
                                "note": "",
                            })

                return data

            # TOON 코드블록·접두어 제거
            cleaned_raw = re.sub(r'```(?:toon)?\n?', '', answer)
            cleaned_raw = re.sub(r'\n?```', '', cleaned_raw).strip()
            if re.match(r'TOON\s*:', cleaned_raw, re.IGNORECASE):
                cleaned_raw = cleaned_raw.split(":", 1)[1].strip()

            # DB 문서의 실제 조리도구 (LLM 환각 방지용)
            doc_tools = []
            for doc in documents:
                tools = doc.metadata.get("cooking_tools", [])
                if tools:
                    doc_tools = tools
                    break

            allergies_list = user_constraints.get("allergies", []) if user_constraints else []

            try:
                recipe = _parse_toon(cleaned_raw)

                r_title     = recipe.get("title", "")
                r_cook_time = recipe.get("cook_time", "")
                r_level     = recipe.get("level", "")
                r_servings  = recipe.get("servings", f"{servings}인분")
                r_intro     = recipe.get("intro", "")
                r_ings      = recipe.get("ingredients", [])
                r_tools     = recipe.get("cooking_tools", "")

                # 알레르기 재료 필터링
                if allergies_list and isinstance(r_ings, list):
                    filtered = []
                    for ing in r_ings:
                        name = ing.get("name", "")
                        matched = [a for a in allergies_list if a in name]
                        if matched:
                            print(f"   [후처리] 알레르기 재료 제거됨: '{name}' (알레르기: {matched})")
                        else:
                            filtered.append(ing)
                    r_ings = filtered

                # intro 마침표 보장
                if r_intro and not r_intro.endswith('.'):
                    r_intro += '.'

                # ingredients → 텍스트
                if isinstance(r_ings, list):
                    ings_str = ", ".join(
                        f"{i.get('name', '')} {i.get('amount', '')}".strip()
                        for i in r_ings if i.get('name')
                    )
                else:
                    ings_str = str(r_ings)

                # 마크다운 조합
                meta_parts = []
                if r_cook_time:
                    meta_parts.append(f"⏱️ {r_cook_time}")
                if r_level:
                    meta_parts.append(f"📊 {r_level}")
                meta_parts.append(f"👥 {r_servings}")

                out_parts = [
                    f"**[{r_title}]**" if r_title else "**[레시피]**",
                    " | ".join(meta_parts),
                ]
                if r_intro:
                    out_parts.append(f"**소개:** {r_intro}")
                if ings_str:
                    out_parts.append(f"**재료:** {ings_str}")
                # 조리도구: DB 메타데이터에 있을 때만 출력 (LLM 환각 방지)
                if doc_tools:
                    out_parts.append(f"**조리도구:** {r_tools if r_tools else ', '.join(doc_tools)}")

                cleaned_answer = "\n".join(out_parts)
                print(f"   [TOON 파싱 성공]")

            except Exception as toon_err:
                print(f"   [TOON 파싱 실패: {toon_err}] → regex fallback")
                cleaned_answer = answer

                # 조리법 섹션 제거
                for pattern in [r'\n조리법[\s:：]+.*', r'\n\*\*조리법\*\*[\s:：]+.*']:
                    match = re.search(pattern, cleaned_answer, re.DOTALL | re.IGNORECASE)
                    if match:
                        cleaned_answer = cleaned_answer[:match.start()].strip()
                        print(f"   [후처리] 조리법 제거됨")
                        break

                # 알레르기 텍스트 제거
                for pattern in [r'\*알레르기.*?\n', r'알레르기 재료.*?\n', r'비선호 음식.*?\n']:
                    cleaned_answer = re.sub(pattern, '', cleaned_answer, flags=re.IGNORECASE)

                # 볼드 통일
                if re.search(r'소개\s*:', cleaned_answer) and '**소개:**' not in cleaned_answer:
                    cleaned_answer = re.sub(r'(?<!\*)소개\s*:\s*', '**소개:** ', cleaned_answer, count=1)
                if re.search(r'재료\s*:', cleaned_answer) and '**재료:**' not in cleaned_answer:
                    cleaned_answer = re.sub(r'(?<!\*)재료\s*:\s*', '**재료:** ', cleaned_answer, count=1)

                # 소개 정제
                if '**소개:**' in cleaned_answer:
                    intro_match = re.search(r'\*\*소개:\*\*\s*(.+)', cleaned_answer)
                    if intro_match:
                        intro_text = intro_match.group(1).strip()
                        intro_text = re.sub(r'[ᄀ-ᄒ]{2,}', '', intro_text)
                        intro_text = re.sub(r'[:;]\)|:\(|:\)|^^|ㅎㅎ|ㅋㅋ', '', intro_text)
                        for phrase in [r'알려드릴게요[!\s]*', r'드릴게요[!\s]*', r'[~]+',
                                       r'답니다[:\s]*\)', r'레시피를 알려드릴게요', r'소개해드릴게요']:
                            intro_text = re.sub(phrase, '', intro_text)
                        intro_text = re.sub(r'\s+', ' ', intro_text).strip()
                        if intro_text and not intro_text.endswith('.'):
                            intro_text += '.'
                        cleaned_answer = re.sub(r'\*\*소개:\*\*\s*.+', f'**소개:** {intro_text}',
                                                cleaned_answer, count=1)
                        print(f"   [후처리] 소개 정제됨: {intro_text[:50]}...")

                # 재료 형식 정리
                if '**재료:**' in cleaned_answer:
                    parts = cleaned_answer.split('**재료:**')
                    if len(parts) == 2:
                        before_ingredients = parts[0]
                        ingredients_section = parts[1].strip()
                        ingredients_lines = []
                        cooking_tools_line = ""
                        for line in ingredients_section.split('\n'):
                            line = line.strip()
                            if re.match(r'\*{0,2}조리도구\*{0,2}\s*:', line):
                                content = re.split(r'\s*:\s*', line, maxsplit=1)
                                if len(content) > 1 and content[1].strip():
                                    cooking_tools_line = line
                                break
                            elif line and not line.startswith('**'):
                                line = re.sub(r'^[-\*]\s*', '', line)
                                if line:
                                    ingredients_lines.append(line)
                            elif line.startswith('**'):
                                break
                        ingredients_text = ', '.join(ingredients_lines)
                        if allergies_list:
                            individual_ings = [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
                            ingredients_text = ', '.join(
                                ing for ing in individual_ings
                                if not any(a in ing for a in allergies_list)
                            )
                        if cooking_tools_line:
                            cleaned_answer = f"{before_ingredients}**재료:** {ingredients_text}\n{cooking_tools_line}"
                        else:
                            cleaned_answer = f"{before_ingredients}**재료:** {ingredients_text}"
                        print(f"   [후처리] 재료 형식 정리됨 (조리도구: {'있음' if cooking_tools_line else '없음'})")

                # 조리도구 처리
                if '조리도구' in cleaned_answer:
                    if not doc_tools:
                        cleaned_answer = re.sub(r'\n?\*{0,2}조리도구\*{0,2}\s*:.*?(?=\n|$)', '',
                                                cleaned_answer, flags=re.MULTILINE)
                        cleaned_answer = cleaned_answer.strip()
                        print(f"   [후처리] 조리도구 제거됨 (메타데이터 없음 - LLM 환각)")
                    else:
                        cleaned_answer = re.sub(r'\n?\*{0,2}조리도구\*{0,2}\s*:\s*(?:\n|$)', '', cleaned_answer)
                        cleaned_answer = cleaned_answer.strip()
                else:
                    if doc_tools:
                        cleaned_answer = f"{cleaned_answer}\n**조리도구:** {', '.join(doc_tools)}"
                        print(f"   [후처리] 조리도구 보완됨 (메타데이터): {', '.join(doc_tools)}")

            print(f"   생성 완료: {cleaned_answer[:50]}...")
            return {"generation": cleaned_answer}
            
        except Exception as e:
            print(f"   생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return {"generation": "답변 생성에 실패했습니다."}

    # ===== 그래프 구성 =====
    
    def decide_to_generate(state: ChatAgentState) -> Literal["web_search", "generate"]:
        """grade 노드 이후 분기 결정"""
        if state.get("web_search_needed") == "yes":
            return "web_search"
        else:
            return "generate"
    
    workflow = StateGraph(ChatAgentState)

    # ── 모든 노드를 timed_node로 감싸기 ──
    workflow.add_node("rewrite",          timed_node("rewrite",          rewrite_query))
    workflow.add_node("retrieve",         timed_node("retrieve",         retrieve))
    workflow.add_node("check_constraints",timed_node("check_constraints",check_constraints))
    workflow.add_node("grade",            timed_node("grade",            grade_documents))
    workflow.add_node("web_search",       timed_node("web_search",       web_search))
    # workflow.add_node("summarize",        timed_node("summarize",        summarize_web_results))  # 제거: 시간 절약
    workflow.add_node("generate",         timed_node("generate",         generate))

    workflow.set_entry_point("rewrite")

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "check_constraints")
    workflow.add_edge("check_constraints", "grade")

    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {"web_search": "web_search", "generate": "generate"}
    )

    workflow.add_edge("web_search", "generate")  # 직접 연결 (요약 스킵)
    # workflow.add_edge("web_search", "summarize")  # 제거
    # workflow.add_edge("summarize", "generate")    # 제거
    workflow.add_edge("generate", END)
    
    compiled = workflow.compile()

    print("[Agent] Adaptive RAG Agent 생성 완료")
    print(f"[Agent] 검색 엔진: {search_engine}")
    return compiled