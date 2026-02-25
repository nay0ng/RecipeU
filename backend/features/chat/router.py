# backend/features/chat/router.py
"""
Chat Agent WebSocket 라우터 - Adaptive RAG + 레시피 수정
"""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Dict
import json
import asyncio
import time
from langchain_naver import ChatClovaX

from core.websocket import manager
from core.dependencies import get_rag_system
from features.chat.agent import create_chat_agent, _node_timings
from models.mysql_db import create_session, add_chat_message
from utils.intent import detect_chat_intent, Intent, extract_allergy_dislike, extract_ingredients_from_modification

logger = logging.getLogger(__name__)

router = APIRouter()

chat_sessions: Dict[str, dict] = {}


# ─────────────────────────────────────────────
# 토큰 사용량 추적 헬퍼 함수
# ─────────────────────────────────────────────
def print_token_usage(response, context_name: str = "LLM"):
    """LLM 응답에서 실제 토큰 사용량 출력"""
    print(f"\n{'='*60}")
    print(f"[{context_name}] HCX API 토큰 사용량 (실측)")
    print(f"{'='*60}")

    # response_metadata 또는 usage_metadata에서 토큰 정보 추출
    usage = None
    if hasattr(response, 'response_metadata'):
        usage = response.response_metadata.get('token_usage') or response.response_metadata.get('usage')
    elif hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata

    if usage:
        prompt_tokens = usage.get('prompt_tokens') or usage.get('promptTokens') or usage.get('input_tokens', 0)
        completion_tokens = usage.get('completion_tokens') or usage.get('completionTokens') or usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens') or usage.get('totalTokens', 0)

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        print(f"📥 입력 토큰 (prompt):     {prompt_tokens:,} tokens")
        print(f"📤 출력 토큰 (completion): {completion_tokens:,} tokens")
        print(f"📊 총 토큰 (total):        {total_tokens:,} tokens")
    else:
        print(f"⚠️  토큰 사용량 정보를 찾을 수 없습니다.")
        print(f"응답 객체 속성: {dir(response)}")
        if hasattr(response, 'response_metadata'):
            print(f"response_metadata: {response.response_metadata}")

    print(f"{'='*60}\n")


def _print_timing_summary(total_ms: float):
    if not _node_timings:
        return
    logger.info("┌─────────────────────────────────────────┐")
    logger.info("│          Node Timing Summary            │")
    logger.info("├─────────────────────────────────────────┤")
    for name, ms in _node_timings.items():
        bar_len = int(ms / max(max(_node_timings.values()), 1) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        pct = (ms / total_ms * 100) if total_ms > 0 else 0
        sec = ms / 1000
        logger.info(f"│  {name:<18} {bar} {sec:>5.1f}초 ({pct:>4.1f}%) │")
    logger.info("├─────────────────────────────────────────┤")
    total_sec = total_ms / 1000
    logger.info(f"│  {'TOTAL':<18} {'':20} {total_sec:>5.1f}초        │")
    logger.info("└─────────────────────────────────────────┘")
    _node_timings.clear()


async def handle_recipe_modification(websocket: WebSocket, session: Dict, user_input: str):
    """레시피 수정 처리 (기존 레시피를 사용자 요청대로 수정)"""
    logger.info("[WS] 🔧 레시피 수정 모드 시작")
    
    # 히스토리에서 원본 레시피와 이미지 찾기 (최근 레시피 우선)
    original_recipe_content = None
    original_image = None

    # 최근 메시지부터 역순으로 검색
    for msg in reversed(session["messages"]):
        if msg["role"] == "assistant":
            content = msg.get("content", "")
            # "재료:" + 이모지(시간/난이도)가 있으면 레시피로 판단
            if "재료" in content and ("⏱️" in content or "📊" in content):
                original_recipe_content = content
                original_image = msg.get("image", "")
                logger.info(f"[WS] 원본 레시피 발견 (최근)")
                logger.info(f"[WS] 원본 이미지: {original_image[:60] if original_image else '없음'}...")
                break
    
    if not original_recipe_content:
        logger.warning("[WS] 원본 레시피 없음 → 일반 대화로 처리")
        return False
    
    await websocket.send_json({"type": "thinking"})

    # 개인화 정보 (알레르기/비선호) 가져오기
    user_constraints = session.get("user_constraints", {})
    allergies = user_constraints.get("allergies", [])
    dislikes = user_constraints.get("dislikes", [])
    constraint_text = ""
    if allergies:
        constraint_text += f"\n- 알레르기 재료 (절대 사용 금지): {', '.join(allergies)}"
    if dislikes:
        constraint_text += f"\n- 비선호 재료 (가능한 피하기): {', '.join(dislikes)}"

    modification_prompt = f"""원본 레시피:
{original_recipe_content}

요청: {user_input}

**개인화 제약사항:**{constraint_text if constraint_text else " 없음"}

**규칙:**
1. 위 레시피만 수정
2. 재료 제거: "A 빼줘" → A 완전 제거
3. 재료 대체: "A 말고 B" → A를 B로 교체
4. 재료 추가: "C 추가" → C 추가 (정확한 양)
5. **알레르기 재료는 수정 결과에 절대 포함 금지**
6. **재료 형식: 쉼표 구분, 한 줄, 줄바꿈 금지**
7. **재료 양 필수 (금지: 약간, 적당량, 조금)**
8. **소개: 객관적, 포멀 (금지: 이모티콘, "~", "답니다:)")**
9. 조리법 출력 금지

**형식 (제목에 반드시 [ ] 포함):**
[변경 사항 1줄]

**[제목]**
⏱️ 시간 | 📊 난이도 | 👥 인분
**소개:** 객관적 1줄 (이모티콘 금지, 포멀하게)
**재료:** 재료1 양, 재료2 양 (한 줄, 쉼표 구분)

**올바른 소개 예시:**
"딸기와 생크림을 활용한 디저트 케이크."
"김치와 돼지고기를 활용한 찌개 요리."

**잘못된 소개 (금지):**
"쫄깃한 면발에 시원한 육수가 별미인 냉우동 입니다."
"대중적인 레시피를 알려드릴게요 ᄒ.ᄒ"

답변:"""
    
    llm = ChatClovaX(model="HCX-003", temperature=0.2, max_tokens=800)

    try:
        result = llm.invoke(modification_prompt)
        modified_recipe = result.content.strip()

        # 후처리: 재료 형식 정리 및 애매한 표현 제거
        import re

        # 볼드 없는 형식을 볼드 형식으로 통일
        if '소개:' in modified_recipe and '**소개:**' not in modified_recipe:
            modified_recipe = re.sub(r'(?<!\*)소개:\s*', '**소개:** ', modified_recipe, count=1)
        if '재료:' in modified_recipe and '**재료:**' not in modified_recipe:
            modified_recipe = re.sub(r'(?<!\*)재료:\s*', '**재료:** ', modified_recipe, count=1)

        # 재료 형식 정리: 줄바꿈 제거, 쉼표로 변환
        if '**재료:**' in modified_recipe:
            parts = modified_recipe.split('**재료:**')
            if len(parts) == 2:
                before_ingredients = parts[0]
                ingredients_section = parts[1].strip()

                ingredients_lines = []
                for line in ingredients_section.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('**'):
                        line = re.sub(r'^[-\*]\s*', '', line)
                        if line:
                            ingredients_lines.append(line)
                    elif line.startswith('**'):
                        break

                ingredients_text = ', '.join(ingredients_lines)
                modified_recipe = f"{before_ingredients}**재료:** {ingredients_text}"
                logger.info("[WS] 재료 형식 정리 완료")

        # 소개 문구 정제
        if '**소개:**' in modified_recipe:
            intro_match = re.search(r'\*\*소개:\*\*\s*(.+)', modified_recipe)
            if intro_match:
                intro_text = intro_match.group(1).strip()

                # 이모티콘 제거 (ᄒ.ᄒ, ᄏᄏ, :), ^^, 등)
                intro_text = re.sub(r'[ᄀ-ᄒ]{2,}', '', intro_text)
                intro_text = re.sub(r'[:;]\)|:\(|:\)|^^|ㅎㅎ|ㅋㅋ', '', intro_text)

                # 캐주얼 표현 제거
                casual_phrases = [
                    r'알려드릴게요[!\s]*', r'드릴게요[!\s]*', r'[~]+', r'요[~]+',
                    r'답니다[:\s]*\)', r'하죠[!\s]*', r'그만큼.*?있답니다',
                    r'레시피를 알려드릴게요', r'소개해드릴게요',
                ]
                for phrase in casual_phrases:
                    intro_text = re.sub(phrase, '', intro_text)

                # 다중 공백 정리
                intro_text = re.sub(r'\s+', ' ', intro_text).strip()
                if intro_text and not intro_text.endswith('.'):
                    intro_text += '.'

                # 소개 문구 교체 (같은 줄만, DOTALL 사용하지 않음)
                modified_recipe = re.sub(
                    r'\*\*소개:\*\*\s*.+',
                    f'**소개:** {intro_text}',
                    modified_recipe,
                    count=1
                )
                logger.info(f"[WS] 소개 정제됨: {intro_text[:50]}...")

        logger.info("[WS] 레시피 수정 완료")

        # 수정사항 파싱 및 modification_history에 추가
        modification_entry = {
            "request": user_input,
            "timestamp": time.time()
        }

        # 간단한 패턴으로 수정 타입 추출 (순서 중요! replace를 먼저 체크)
        # "A 말고 B 넣어줘" 같은 패턴은 replace
        if any(kw in user_input for kw in ["대신", "말고", "바꿔", "교체"]) and any(kw in user_input for kw in ["추가", "넣어", "로"]):
            modification_entry["type"] = "replace"
        elif any(kw in user_input for kw in ["빼", "제거", "없이", "없어", "없는", "없다"]):
            modification_entry["type"] = "remove"
        elif any(kw in user_input for kw in ["대신", "바꿔", "교체", "말고"]):
            modification_entry["type"] = "replace"
        elif any(kw in user_input for kw in ["추가", "넣어"]):
            modification_entry["type"] = "add"
        else:
            modification_entry["type"] = "modify"

        # 재료명 추출 (remove/replace/add 타입)
        if modification_entry["type"] in ["remove", "replace", "add"]:
            extracted = extract_ingredients_from_modification(user_input, modification_entry["type"])
            modification_entry["remove_ingredients"] = extracted.get("remove", [])
            modification_entry["add_ingredients"] = extracted.get("add", [])
            logger.info(f"[WS] 🔍 추출된 재료 - 제거: {extracted.get('remove', [])}, 추가: {extracted.get('add', [])}")
        else:
            modification_entry["remove_ingredients"] = []
            modification_entry["add_ingredients"] = []

        if "modification_history" not in session:
            session["modification_history"] = []
        session["modification_history"].append(modification_entry)

        logger.info(f"[WS] 수정 이력 추가: type={modification_entry['type']}, request='{modification_entry['request']}', remove={modification_entry.get('remove_ingredients', [])}, add={modification_entry.get('add_ingredients', [])}")
        logger.info(f"[WS] 현재 누적 수정 이력 ({len(session['modification_history'])}개):")
        for i, mod in enumerate(session["modification_history"], 1):
            logger.info(f"     [{i}] type={mod.get('type')}, remove={mod.get('remove_ingredients', [])}, add={mod.get('add_ingredients', [])}")
        # 히스토리에 추가 (이미지 포함!)
        session["messages"].append({
            "role": "assistant",
            "content": modified_recipe,
            "image": original_image  # 원본 이미지 유지
        })

        # WebSocket 응답 (이미지 포함 + hideImage + modification_history)
        await websocket.send_json({
            "type": "agent_message",
            "content": modified_recipe,
            "image": original_image,  # 데이터 전달
            "hideImage": True,  # UI에는 안 보이게
            "modification_history": session["modification_history"]  # 누적 수정 이력 전달
        })

        return True
        
    except Exception as e:
        logger.error(f"[WS] ❌ 레시피 수정 실패: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": "레시피 수정 중 오류가 발생했습니다."
        })
        return True


@router.websocket("/ws/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: str,
    rag_system = Depends(get_rag_system),
):
    await websocket.accept()
    logger.info(f"[WS] Connected: {session_id}")

    if not rag_system:
        logger.warning("[WS] RAG 시스템 없음")
        await websocket.send_json({"type": "error", "message": "RAG 시스템을 사용할 수 없습니다."})
        await websocket.close()
        return

    try:
        agent = create_chat_agent(rag_system)
        if not agent:
            raise ValueError("Agent 생성 실패")
        logger.info("[WS] Adaptive RAG Agent 생성 완료")
    except Exception as e:
        logger.error(f"[WS] Agent 생성 에러: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Agent 생성 실패: {str(e)}"})
        await websocket.close()
        return

    manager.active_connections[session_id] = websocket

    # DB 세션은 init_context에서 member_id를 받은 후 생성
    db_session_id = None
    member_id = 0  # 기본값, init_context에서 업데이트

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "messages": [],
            "user_constraints": {},
            "last_documents": [],
            "last_agent_response": "",
            "db_session_id": None,
            "member_id": 0,
            "temp_allowed_dislikes": [],  # 세션 내 임시 허용된 비선호 음식
            "modification_history": [],  # 레시피 수정 이력 (누적)
        }

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            logger.info(f"[WS] 메시지 수신: {msg_type}")

            if msg_type == "init_context":
                member_info = message.get("member_info", {})
                initial_history = message.get("initial_history", [])
                modification_history = message.get("modification_history", [])  # ✅ 수정 이력 받기

                chat_sessions[session_id]["user_constraints"] = member_info

                # 수정 이력 복원 (재생성으로 돌아온 경우)
                if modification_history:
                    chat_sessions[session_id]["modification_history"] = modification_history
                    logger.info(f"[WS] 🔄 수정 이력 복원: {len(modification_history)}개")
                    for i, mod in enumerate(modification_history, 1):
                        logger.info(f"     [{i}] type={mod.get('type')}, request='{mod.get('request')}'")

                # member_id 추출 및 DB 세션 생성
                mid = member_info.get("member_id")
                logger.info(f"[WS] init_context 수신: member_id={mid} (type: {type(mid).__name__})")

                # member_id를 int로 변환 (숫자 또는 숫자 문자열 모두 처리)
                try:
                    member_id = int(mid) if mid is not None else 0
                except (ValueError, TypeError):
                    member_id = 0

                if member_id > 0:
                    chat_sessions[session_id]["member_id"] = member_id

                    # DB 세션이 아직 없으면 생성
                    if not chat_sessions[session_id].get("db_session_id"):
                        try:
                            from models.mysql_db import create_session
                            db_result = create_session(member_id=member_id)
                            db_session_id = db_result.get("session_id") if db_result else None
                            chat_sessions[session_id]["db_session_id"] = db_session_id

                            # 클라이언트로 db_session_id 전송
                            if db_session_id:
                                await websocket.send_json({
                                    "type": "session_initialized",
                                    "session_id": session_id,
                                    "db_session_id": db_session_id
                                })
                                logger.info(f"[WS] DB 세션 생성 완료: db_session_id={db_session_id}, member_id={member_id}")
                            else:
                                logger.warning(f"[WS] DB 세션 생성 결과가 None: db_result={db_result}")
                        except Exception as e:
                            logger.error(f"[WS] DB 세션 생성 실패: {e}", exc_info=True)
                else:
                    logger.warning(f"[WS] member_id가 0 또는 유효하지 않음: {mid}")

                # 초기 히스토리 설정 (레시피 수정 모드용)
                if initial_history:
                    chat_sessions[session_id]["messages"].extend(initial_history)
                    logger.info(f"[WS] 초기 히스토리 {len(initial_history)}개 추가")

                logger.info(f"[WS] 컨텍스트 설정: {member_info.get('names', [])}, member_id={member_id}")
                continue

            elif msg_type == "constraint_confirmation":
                # 제약사항 충돌 확인 응답 처리
                confirmation = message.get("confirmation")  # "yes" or "no"
                logger.info(f"[WS] 제약사항 충돌 확인 응답: {confirmation}")

                start_time = time.time()

                if confirmation == "no":
                    # 거절 → 다른 레시피 제안
                    reject_msg = "알겠습니다. 다른 레시피를 검색해드릴까요? 또는 기존 레시피를 수정해드릴 수도 있습니다."
                    chat_sessions[session_id]["messages"].append({
                        "role": "assistant",
                        "content": reject_msg
                    })
                    chat_sessions[session_id].pop("pending_constraint_search", None)

                    await websocket.send_json({
                        "type": "agent_message",
                        "content": reject_msg
                    })
                    logger.info("[WS] 제약사항 충돌 거절 → 다른 레시피 제안")
                    continue

                elif confirmation == "yes":
                    # 승인 → pending_constraint_search로 레시피 생성 진행 & 제약사항에서 제거
                    pending = chat_sessions[session_id].get("pending_constraint_search")
                    if not pending:
                        logger.warning("[WS] pending_constraint_search가 없음")
                        await websocket.send_json({
                            "type": "error",
                            "message": "이전 검색 정보를 찾을 수 없습니다."
                        })
                        continue

                    content = pending["query"]
                    conflicted_ingredients = pending.get("conflicted_ingredients", [])

                    logger.info(f"[WS] 제약사항 충돌 승인 → 레시피 생성 진행: {content}")
                    logger.info(f"[WS] 제약사항에서 제거할 재료: {conflicted_ingredients}")

                    # modification_history에서 충돌 재료 제거 (이번 세션에서만)
                    modification_history = chat_sessions[session_id].get("modification_history", [])
                    updated_history = []
                    for mod in modification_history:
                        if mod.get("type") in ["remove", "replace"]:
                            # 충돌 재료를 remove_ingredients에서 제거
                            remaining_remove = [
                                ing for ing in mod.get("remove_ingredients", [])
                                if ing not in conflicted_ingredients
                            ]
                            # 재료가 남아있으면 유지, 없으면 제거
                            if remaining_remove or not mod.get("remove_ingredients"):
                                mod["remove_ingredients"] = remaining_remove
                                updated_history.append(mod)
                            else:
                                logger.info(f"[WS] 수정 이력 제거: {mod['request']} (제거할 재료 모두 삭제됨)")
                        else:
                            updated_history.append(mod)

                    chat_sessions[session_id]["modification_history"] = updated_history
                    logger.info(f"[WS] 업데이트된 수정 이력: {len(updated_history)}개")

                    # pending_constraint_search 정리
                    chat_sessions[session_id].pop("pending_constraint_search", None)

                    # 레시피 검색 진행 (아래의 레시피 검색 모드 로직으로 점프)
                    logger.info(f"[WS] 레시피 검색 모드 시작 (제약사항 충돌 승인 후)")

                    chat_history = [
                        f"{msg['role']}: {msg['content']}"
                        for msg in chat_sessions[session_id]["messages"]
                    ]

                    await websocket.send_json({"type": "thinking", "message": "레시피 검색 중..."})

                    # 업데이트된 수정 이력 전달
                    modification_history = chat_sessions[session_id].get("modification_history", [])
                    logger.info(f"[WS] 수정 이력 전달: {len(modification_history)}개")
                    if modification_history:
                        for i, mod in enumerate(modification_history, 1):
                            logger.info(f"     [{i}] type={mod.get('type')}, remove={mod.get('remove_ingredients', [])}, add={mod.get('add_ingredients', [])}")

                    agent_state = {
                        "question": content,
                        "original_question": content,
                        "chat_history": chat_history,
                        "documents": [],
                        "generation": "",
                        "web_search_needed": "no",
                        "user_constraints": chat_sessions[session_id]["user_constraints"],
                        "constraint_warning": "",
                        "modification_history": modification_history
                    }

                    async def progress_notifier():
                        steps = [
                            (0, "쿼리 재작성 중..."),
                            (3, "레시피 검색 중..."),
                            (6, "관련성 평가 중..."),
                            (10, "답변 생성 중..."),
                            (15, "거의 완료...")
                        ]
                        for delay, msg in steps:
                            await asyncio.sleep(delay if delay == 0 else 3)
                            if time.time() - start_time < 20:
                                await websocket.send_json({
                                    "type": "progress",
                                    "message": f"{msg} ({int(time.time() - start_time)}초)"
                                })
                            else:
                                break

                    notifier_task = asyncio.create_task(progress_notifier())

                    try:
                        _node_timings.clear()

                        async def run_agent():
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(None, agent.invoke, agent_state)

                        result = await asyncio.wait_for(run_agent(), timeout=20.0)

                        total_ms = (time.time() - start_time) * 1000
                        _print_timing_summary(total_ms)

                        # 캐시 저장
                        agent_docs = result.get("documents", [])
                        agent_response = result.get("generation", "")

                        if agent_docs:
                            chat_sessions[session_id]["last_documents"] = [
                                {
                                    "content": doc.page_content,
                                    "title": doc.metadata.get("title", ""),
                                    "cook_time": doc.metadata.get("cook_time", ""),
                                    "level": doc.metadata.get("level", ""),
                                    "recipe_id": doc.metadata.get("recipe_id", ""),
                                }
                                for doc in agent_docs
                            ]
                            logger.info(f"[WS] 세션 캐시 저장: {len(agent_docs)}개 문서")

                        if agent_response:
                            chat_sessions[session_id]["last_agent_response"] = agent_response
                            logger.info(f"[WS] Agent 답변 캐시: {agent_response[:60]}...")

                        response = agent_response or "답변을 생성할 수 없습니다."

                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": response
                        })

                        await websocket.send_json({
                            "type": "agent_message",
                            "content": response
                        })

                        total_sec = total_ms / 1000
                        logger.info(f"[WS] 응답 완료 (총 {total_sec:.1f}초)")

                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time
                        logger.warning(f"[WS] ⏱Agent 타임아웃 ({elapsed:.1f}초)")
                        _print_timing_summary(elapsed * 1000)

                        await websocket.send_json({
                            "type": "agent_message",
                            "content": f"죄송합니다. 응답 시간이 너무 오래 걸렸어요 ({int(elapsed)}초). 다시 시도해주세요."
                        })

                    except Exception as e:
                        elapsed = time.time() - start_time
                        logger.error(f"[WS] Agent 실행 에러 ({elapsed:.1f}초): {e}", exc_info=True)
                        _print_timing_summary(elapsed * 1000)

                        await websocket.send_json({
                            "type": "error",
                            "message": f"오류가 발생했습니다 ({int(elapsed)}초). 다시 시도해주세요."
                        })

                    finally:
                        notifier_task.cancel()
                        try:
                            await notifier_task
                        except asyncio.CancelledError:
                            pass

                    # 레시피 생성 완료 후 다음 메시지 처리를 위해 continue
                    continue

                else:
                    logger.warning(f"[WS] 알 수 없는 confirmation 값: {confirmation}")
                    continue

            elif msg_type == "allergy_confirmation":
                # 알러지 경고 확인 응답 처리
                confirmation = message.get("confirmation")  # "yes" or "no"
                logger.info(f"[WS] 알러지 확인 응답: {confirmation}")

                start_time = time.time()

                if confirmation == "no":
                    # 거절 → 다른 레시피 제안
                    reject_msg = "알겠습니다. 다른 레시피를 검색해드릴까요?"
                    chat_sessions[session_id]["messages"].append({
                        "role": "assistant",
                        "content": reject_msg
                    })
                    chat_sessions[session_id].pop("pending_search", None)

                    await websocket.send_json({
                        "type": "agent_message",
                        "content": reject_msg
                    })
                    logger.info("[WS] 알러지 경고 거절 → 다른 레시피 제안")
                    continue

                elif confirmation == "yes":
                    # 승인 → pending_search로 레시피 생성 진행
                    pending = chat_sessions[session_id].get("pending_search")
                    if not pending:
                        logger.warning("[WS] pending_search가 없음")
                        await websocket.send_json({
                            "type": "error",
                            "message": "이전 검색 정보를 찾을 수 없습니다."
                        })
                        continue

                    content = pending["query"]
                    matched_dislikes = pending.get("matched_dislikes", [])

                    logger.info(f"[WS] 비선호 음식 경고 승인 → 레시피 생성 진행: {content}")
                    logger.info(f"[WS] 임시 제외할 비선호: {matched_dislikes}")

                    # 세션에 임시 허용 목록 추가 (이번 세션 내에서만 유효)
                    if "temp_allowed_dislikes" not in chat_sessions[session_id]:
                        chat_sessions[session_id]["temp_allowed_dislikes"] = []
                    chat_sessions[session_id]["temp_allowed_dislikes"].extend(matched_dislikes)
                    chat_sessions[session_id]["temp_allowed_dislikes"] = list(set(chat_sessions[session_id]["temp_allowed_dislikes"]))

                    logger.info(f"[WS] 세션 내 임시 허용된 비선호: {chat_sessions[session_id]['temp_allowed_dislikes']}")

                    # user_constraints에서 매칭된 비선호 임시 제거 (사용자가 "예"를 눌렀으므로)
                    # B는 변경하지 않음! 이번 검색에만 임시로 제거
                    original_constraints = chat_sessions[session_id]["user_constraints"]
                    modified_constraints = original_constraints.copy()

                    # 비선호 목록에서 매칭된 항목만 임시 제거
                    if "dislikes" in modified_constraints:
                        modified_constraints["dislikes"] = [
                            item for item in modified_constraints["dislikes"]
                            if item not in matched_dislikes
                        ]

                    logger.info(f"[WS] 임시 수정된 제약 조건: allergies={modified_constraints.get('allergies', [])}, dislikes={modified_constraints.get('dislikes', [])}")

                    # pending_search 정리
                    chat_sessions[session_id].pop("pending_search", None)

                    # 레시피 검색 진행 (아래의 레시피 검색 모드 로직으로 점프)
                    # 레시피 검색 모드 (RAG 사용)
                    logger.info(f"[WS] 레시피 검색 모드 시작 (알러지 승인 후)")

                    chat_history = [
                        f"{msg['role']}: {msg['content']}"
                        for msg in chat_sessions[session_id]["messages"]
                    ]

                    await websocket.send_json({"type": "thinking", "message": "레시피 검색 중..."})

                    # 수정 이력 가져오기
                    modification_history = chat_sessions[session_id].get("modification_history", [])
                    logger.info(f"[WS] 🔧 수정 이력 전달: {len(modification_history)}개")
                    if modification_history:
                        for i, mod in enumerate(modification_history, 1):
                            logger.info(f"     [{i}] type={mod.get('type')}, request='{mod.get('request')}'")

                    agent_state = {
                        "question": content,
                        "original_question": content,
                        "chat_history": chat_history,
                        "documents": [],
                        "generation": "",
                        "web_search_needed": "no",
                        "user_constraints": modified_constraints, 
                        "constraint_warning": "",
                        "modification_history": modification_history 
                    }

                    async def progress_notifier():
                        steps = [
                            (0, "쿼리 재작성 중..."),
                            (3, "레시피 검색 중..."),
                            (6, "관련성 평가 중..."),
                            (10, "답변 생성 중..."),
                            (15, "거의 완료...")
                        ]
                        for delay, msg in steps:
                            await asyncio.sleep(delay if delay == 0 else 3)
                            if time.time() - start_time < 20:
                                await websocket.send_json({
                                    "type": "progress",
                                    "message": f"{msg} ({int(time.time() - start_time)}초)"
                                })
                            else:
                                break

                    notifier_task = asyncio.create_task(progress_notifier())

                    try:
                        _node_timings.clear()

                        async def run_agent():
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(None, agent.invoke, agent_state)

                        result = await asyncio.wait_for(run_agent(), timeout=20.0)

                        total_ms = (time.time() - start_time) * 1000
                        _print_timing_summary(total_ms)

                        # 캐시 저장
                        agent_docs = result.get("documents", [])
                        agent_response = result.get("generation", "")

                        if agent_docs:
                            chat_sessions[session_id]["last_documents"] = [
                                {
                                    "content": doc.page_content,
                                    "title": doc.metadata.get("title", ""),
                                    "cook_time": doc.metadata.get("cook_time", ""),
                                    "level": doc.metadata.get("level", ""),
                                    "recipe_id": doc.metadata.get("recipe_id", ""),
                                }
                                for doc in agent_docs
                            ]
                            logger.info(f"[WS] 세션 캐시 저장: {len(agent_docs)}개 문서")

                        if agent_response:
                            chat_sessions[session_id]["last_agent_response"] = agent_response
                            logger.info(f"[WS] Agent 답변 캐시: {agent_response[:60]}...")

                        response = agent_response or "답변을 생성할 수 없습니다."

                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": response
                        })

                        await websocket.send_json({
                            "type": "agent_message",
                            "content": response
                        })

                        total_sec = total_ms / 1000
                        logger.info(f"[WS] 응답 완료 (총 {total_sec:.1f}초)")

                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time
                        logger.warning(f"[WS] ⏱Agent 타임아웃 ({elapsed:.1f}초)")
                        _print_timing_summary(elapsed * 1000)

                        await websocket.send_json({
                            "type": "agent_message",
                            "content": f"죄송합니다. 응답 시간이 너무 오래 걸렸어요 ({int(elapsed)}초). 다시 시도해주세요."
                        })

                    except Exception as e:
                        elapsed = time.time() - start_time
                        logger.error(f"[WS] Agent 실행 에러 ({elapsed:.1f}초): {e}", exc_info=True)
                        _print_timing_summary(elapsed * 1000)

                        await websocket.send_json({
                            "type": "error",
                            "message": f"오류가 발생했습니다 ({int(elapsed)}초). 다시 시도해주세요."
                        })

                    finally:
                        notifier_task.cancel()
                        try:
                            await notifier_task
                        except asyncio.CancelledError:
                            pass

                    # 레시피 생성 완료 후 다음 메시지 처리를 위해 continue
                    continue

                else:
                    logger.warning(f"[WS] 알 수 없는 confirmation 값: {confirmation}")
                    continue

            elif msg_type == "user_message":
                content = message.get("content", "")

                logger.info(f"[WS] 사용자 메시지: {content}")

                start_time = time.time()

                # AI Safety 필터 (의도 분류 전에 먼저 처리)
                # PII(개인정보) + 욕설/비속어 + 유해 입력 감지
                try:
                    from langchain_core.messages import HumanMessage as _HM
                    safety_llm = ChatClovaX(model="HCX-DASH-001", temperature=0.1, max_tokens=10)
                    safety_result = safety_llm.invoke([_HM(content=f"""입력의 유해성을 판단하세요.

입력: "{content}"

**감지 대상:**
1. PII(개인정보): 전화번호, 주민번호, 이메일, 주소, 계좌번호, 비밀번호, 이름+연락처 조합
2. 욕설/비속어: 비하, 혐오, 성적 표현, 폭언
3. 유해 요청: 폭력, 불법 행위, 차별, 자해 관련

**예외 (감지하지 않음):**
- 음식 재료명, 요리명, 조리 용어
- 단순 이름만("홍길동")은 PII 아님

유해하면 YES, 안전하면 NO (한 단어만):""")])
                    safety_answer = safety_result.content.strip().upper()
                    logger.info(f"[WS] AI Safety 감지 LLM 응답: {safety_answer}")

                    if "YES" in safety_answer:
                        logger.info(f"[WS] AI Safety 감지 → 차단: {content[:50]}")
                        block_msg = "해당 내용에는 응답할 수 없습니다. 적절한 내용으로 다시 질문해주세요."

                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": block_msg
                        })

                        await websocket.send_json({
                            "type": "safety_block",
                            "content": block_msg
                        })

                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] AI Safety 차단 완료 (총 {total_sec:.1f}초)")
                        continue

                except Exception as e:
                    logger.warning(f"[WS] AI Safety 감지 실패 (무시하고 진행): {e}")

                # 사용자 메시지 히스토리에 추가
                chat_sessions[session_id]["messages"].append({
                    "role": "user",
                    "content": content
                })

                # 의도 분류
                user_intent = detect_chat_intent(content, chat_sessions[session_id]["messages"])
                logger.info(f"[WS] 의도 분류: {user_intent}")

                # 알러지/비선호 감지 (회원만, 레시피 검색/수정이 아닐 때만)
                member_id = chat_sessions[session_id].get("member_id", 0)
                if member_id > 0 and user_intent not in [Intent.RECIPE_SEARCH, Intent.RECIPE_MODIFY]:
                    # chat_history를 전달하여 레시피 존재 여부 확인
                    allergy_dislike_data = extract_allergy_dislike(
                        content,
                        chat_history=chat_sessions[session_id]["messages"]
                    )
                    if allergy_dislike_data.get("type"):
                        detected_type = allergy_dislike_data["type"]
                        detected_items = allergy_dislike_data["items"]

                        logger.info(f"[WS] 알러지/비선호 감지: type={detected_type}, items={detected_items}")

                        # 간단한 응답 + 버튼 데이터 전송
                        if detected_items:
                            response_msg = f"알겠습니다. 앞으로 레시피 추천 시 참고하겠습니다."
                        else:
                            response_msg = "알겠습니다."

                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": response_msg
                        })

                        # WebSocket 응답 (버튼 포함)
                        await websocket.send_json({
                            "type": "allergy_dislike_detected",
                            "content": response_msg,
                            "detected_type": detected_type,
                            "detected_items": detected_items,
                            "show_button": True if detected_items else False
                        })

                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] 알러지/비선호 감지 완료 (총 {total_sec:.1f}초)")
                        continue

                # 1. 요리 무관 질문 → title 검색으로 재확인 후 외부 챗봇으로 리다이렉트
                if user_intent == Intent.NOT_COOKING:
                    logger.info(f"[WS] 요리 무관 질문 감지 → title 검색으로 재확인")
                    # title 매칭만 사용 (벡터 검색은 아무거나 반환하므로 제외)
                    title_check = rag_system._milvus_title_search(content, k=1)
                    if title_check and len(title_check) > 0:
                        logger.info(f"[WS] title 매칭 결과 있음 → RECIPE_SEARCH로 변경 (줄임말/신조어 가능)")
                        user_intent = Intent.RECIPE_SEARCH
                    else:
                        logger.info(f"[WS] RAG 검색 결과 없음 → 외부 챗봇 리다이렉트")
                        redirect_msg = "레시피 외의 질문은 외부 챗봇을 이용해 주세요."

                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": redirect_msg
                        })

                        await websocket.send_json({
                            "type": "chat_external",
                            "content": redirect_msg
                        })

                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] 외부 챗봇 리다이렉트 (총 {total_sec:.1f}초)")
                        continue

                # 2. 요리 관련 질문 → LLM 답변 (레시피 없이)
                if user_intent == Intent.COOKING_QUESTION:
                    logger.info(f"[WS] 요리 관련 질문 처리")
                    await websocket.send_json({"type": "thinking"})

                    # 대화 히스토리 포함
                    chat_history_text = "\n".join([
                        f"{msg['role']}: {msg['content'][:200]}"
                        for msg in chat_sessions[session_id]["messages"][-5:]
                    ])

                    question_prompt = f"""요리 전문가로서 질문에 답변하세요.

대화 맥락:
{chat_history_text}

질문: {content}

**규칙:**
1. 간결하고 명확하게 답변 (2-3문장)
2. 구체적인 팁이나 대안 제시
3. 포멀하고 전문적인 톤
4. 개인정보(이름, 전화번호, 주소, 이메일 등) 포함 질문에는 응답 거부
5. 요리/음식과 무관한 질문에는 응답하지 않음

답변:"""

                    try:
                        llm = ChatClovaX(model="HCX-DASH-001", temperature=0.2, max_tokens=200)
                        result = llm.invoke(question_prompt)
                        answer = result.content.strip()

                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": answer
                        })

                        await websocket.send_json({
                            "type": "agent_message",
                            "content": answer
                        })

                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] 요리 질문 답변 완료 (총 {total_sec:.1f}초)")
                        continue

                    except Exception as e:
                        logger.error(f"[WS] 요리 질문 답변 실패: {e}")
                        # 실패 시 레시피 검색으로 폴백
                        logger.info("[WS] 레시피 검색으로 전환")

                # 3. 레시피 수정 모드 처리
                if user_intent == Intent.RECIPE_MODIFY:
                    # 알레르기 재료 체크 (추가/대체 요청에 알레르기 재료가 포함되어 있는지)
                    user_constraints = chat_sessions[session_id].get("user_constraints", {})
                    user_allergies = user_constraints.get("allergies", [])

                    if user_allergies:
                        content_lower = content.lower()
                        matched_allergies = [item for item in user_allergies if item in content_lower]

                        if matched_allergies:
                            allergy_block_msg = f"알레르기 재료({', '.join(matched_allergies)})가 포함되어 있어 해당 수정을 진행할 수 없습니다. 다른 재료로 변경해주세요."
                            logger.info(f"[WS] 레시피 수정 시 알레르기 재료 감지 → 차단: {matched_allergies}")

                            chat_sessions[session_id]["messages"].append({
                                "role": "assistant",
                                "content": allergy_block_msg
                            })

                            await websocket.send_json({
                                "type": "allergy_block",
                                "content": allergy_block_msg,
                                "matched_allergies": matched_allergies
                            })

                            total_sec = (time.time() - start_time)
                            logger.info(f"[WS] 알레르기 차단 완료 (총 {total_sec:.1f}초)")
                            continue

                    modification_success = await handle_recipe_modification(
                        websocket,
                        chat_sessions[session_id],
                        content
                    )

                    if modification_success:
                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] 레시피 수정 완료 (총 {total_sec:.1f}초)")
                        continue

                    logger.info("[WS] 수정 실패 → 레시피 검색으로 전환")

                # 4. 레시피 검색 모드 (RAG 사용)
                logger.info(f"[WS] 레시피 검색 모드 시작 (의도: {user_intent})")

                # 수정 이력의 제약사항과 충돌 체크 (모든 사용자)
                if user_intent == Intent.RECIPE_SEARCH:
                    modification_history = chat_sessions[session_id].get("modification_history", [])

                    # remove/replace 타입에서 제거할 재료(remove_ingredients)만 수집
                    constrained_ingredients = []
                    for mod in modification_history:
                        if mod.get("type") in ["remove", "replace"]:
                            constrained_ingredients.extend(mod.get("remove_ingredients", []))

                    # 중복 제거
                    constrained_ingredients = list(set(constrained_ingredients))

                    if constrained_ingredients:
                        # 검색어에 제약 재료가 포함되어 있는지 확인
                        content_lower = content.lower()
                        conflicted_ingredients = [
                            ing for ing in constrained_ingredients
                            if ing in content_lower
                        ]

                        if conflicted_ingredients:
                            warning_msg = f"{', '.join(conflicted_ingredients)}은(는) 이전에 사용자님이 제외하신 재료입니다. 괜찮으신가요?"

                            logger.info(f"[WS] 제약사항 충돌 감지: {conflicted_ingredients}")

                            # pending_constraint_search 상태 저장
                            chat_sessions[session_id]["pending_constraint_search"] = {
                                "query": content,
                                "conflicted_ingredients": conflicted_ingredients
                            }

                            # 히스토리에 경고 메시지 추가
                            chat_sessions[session_id]["messages"].append({
                                "role": "assistant",
                                "content": warning_msg
                            })

                            # WebSocket으로 경고 + 확인 버튼 전송
                            await websocket.send_json({
                                "type": "constraint_warning",
                                "content": warning_msg,
                                "conflicted_ingredients": conflicted_ingredients,
                                "show_confirmation": True
                            })

                            total_sec = (time.time() - start_time)
                            logger.info(f"[WS] 제약사항 충돌 확인 요청 완료 (총 {total_sec:.1f}초)")
                            continue

                # 알러지/비선호 재료가 포함된 검색인지 확인 (회원만)
                if user_intent == Intent.RECIPE_SEARCH and member_id > 0:
                    user_constraints = chat_sessions[session_id].get("user_constraints", {})
                    user_allergies = user_constraints.get("allergies", [])
                    user_dislikes = user_constraints.get("dislikes", [])

                    # 세션 내 임시 허용된 비선호 음식 가져오기
                    temp_allowed = chat_sessions[session_id].get("temp_allowed_dislikes", [])

                    # 검색어에 알러지/비선호 재료가 포함되어 있는지 확인
                    matched_allergies = [item for item in user_allergies if item in content]
                    # 임시 허용된 비선호는 제외
                    matched_dislikes = [item for item in user_dislikes if item in content and item not in temp_allowed]

                    # 알러지 재료 포함 → 무조건 차단
                    if matched_allergies:
                        allergy_block_msg = f"알러지 재료({', '.join(matched_allergies)})가 포함되어 있어 레시피를 생성할 수 없습니다. 다른 레시피를 검색해주세요."

                        logger.info(f"[WS] 알러지 재료 감지 → 생성 차단: {matched_allergies}")

                        # 히스토리에 차단 메시지 추가
                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": allergy_block_msg
                        })

                        # WebSocket으로 차단 메시지 전송 (버튼 없음)
                        await websocket.send_json({
                            "type": "agent_message",
                            "content": allergy_block_msg
                        })

                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] 알러지 재료 차단 완료 (총 {total_sec:.1f}초)")
                        continue

                    # 비선호 음식만 포함 → 확인 요청
                    if matched_dislikes:
                        warning_msg = f"비선호 음식({', '.join(matched_dislikes)})이(가) 포함되어 있습니다. 그래도 생성해드릴까요?"

                        logger.info(f"[WS] 비선호 음식 감지: {matched_dislikes}")

                        # pending_search 상태 저장 (비선호만 저장)
                        chat_sessions[session_id]["pending_search"] = {
                            "query": content,
                            "user_constraints": user_constraints,
                            "matched_dislikes": matched_dislikes
                        }

                        # 히스토리에 경고 메시지 추가
                        chat_sessions[session_id]["messages"].append({
                            "role": "assistant",
                            "content": warning_msg
                        })

                        # WebSocket으로 경고 + 확인 버튼 전송
                        await websocket.send_json({
                            "type": "allergy_warning",
                            "content": warning_msg,
                            "matched_dislikes": matched_dislikes,
                            "show_confirmation": True
                        })

                        total_sec = (time.time() - start_time)
                        logger.info(f"[WS] 비선호 음식 확인 요청 완료 (총 {total_sec:.1f}초)")
                        continue

                chat_history = [
                    f"{msg['role']}: {msg['content']}"
                    for msg in chat_sessions[session_id]["messages"]
                ]

                await websocket.send_json({"type": "thinking", "message": "레시피 검색 중..."})

                # 수정 이력 가져오기
                modification_history = chat_sessions[session_id].get("modification_history", [])
                logger.info(f"[WS] 수정 이력 전달: {len(modification_history)}개")
                if modification_history:
                    for i, mod in enumerate(modification_history, 1):
                        logger.info(f"     [{i}] type={mod.get('type')}, request='{mod.get('request')}'")

                agent_state = {
                    "question": content,
                    "original_question": content,
                    "chat_history": chat_history,
                    "documents": [],
                    "generation": "",
                    "web_search_needed": "no",
                    "user_constraints": chat_sessions[session_id]["user_constraints"],
                    "constraint_warning": "",
                    "modification_history": modification_history  
                }

                async def progress_notifier():
                    steps = [
                        (0, "쿼리 재작성 중..."), 
                        (3, "레시피 검색 중..."), 
                        (6, "관련성 평가 중..."), 
                        (10, "답변 생성 중..."), 
                        (15, "거의 완료...")
                    ]
                    for delay, msg in steps:
                        await asyncio.sleep(delay if delay == 0 else 3)
                        if time.time() - start_time < 20:
                            await websocket.send_json({
                                "type": "progress", 
                                "message": f"{msg} ({int(time.time() - start_time)}초)"
                            })
                        else:
                            break

                notifier_task = asyncio.create_task(progress_notifier())

                try:
                    _node_timings.clear()

                    async def run_agent():
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, agent.invoke, agent_state)

                    result = await asyncio.wait_for(run_agent(), timeout=20.0)

                    total_ms = (time.time() - start_time) * 1000
                    _print_timing_summary(total_ms)

                    # 캐시 저장
                    agent_docs = result.get("documents", [])
                    agent_response = result.get("generation", "")

                    if agent_docs:
                        chat_sessions[session_id]["last_documents"] = [
                            {
                                "content": doc.page_content,
                                "title": doc.metadata.get("title", ""),
                                "cook_time": doc.metadata.get("cook_time", ""),
                                "level": doc.metadata.get("level", ""),
                                "recipe_id": doc.metadata.get("recipe_id", ""),
                            }
                            for doc in agent_docs
                        ]
                        logger.info(f"[WS] 세션 캐시 저장: {len(agent_docs)}개 문서")

                    if agent_response:
                        chat_sessions[session_id]["last_agent_response"] = agent_response
                        logger.info(f"[WS] Agent 답변 캐시: {agent_response[:60]}...")

                    response = agent_response or "답변을 생성할 수 없습니다."

                    chat_sessions[session_id]["messages"].append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    await websocket.send_json({
                        "type": "agent_message", 
                        "content": response
                    })
                    
                    total_sec = total_ms / 1000
                    logger.info(f"[WS] 응답 완료 (총 {total_sec:.1f}초)")

                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"[WS] Agent 타임아웃 ({elapsed:.1f}초)")
                    _print_timing_summary(elapsed * 1000)
                    
                    await websocket.send_json({
                        "type": "agent_message",
                        "content": f"죄송합니다. 응답 시간이 너무 오래 걸렸어요 ({int(elapsed)}초). 다시 시도해주세요."
                    })
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(f"[WS] Agent 실행 에러 ({elapsed:.1f}초): {e}", exc_info=True)
                    _print_timing_summary(elapsed * 1000)
                    
                    await websocket.send_json({
                        "type": "error", 
                        "message": f"오류가 발생했습니다 ({int(elapsed)}초). 다시 시도해주세요."
                    })
                    
                finally:
                    notifier_task.cancel()
                    try:
                        await notifier_task
                    except asyncio.CancelledError:
                        pass

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"[WS] 에러: {e}", exc_info=True)
    finally:
        manager.disconnect(session_id)
        logger.info(f"[WS] Closed: {session_id}")


@router.get("/session/{session_id}")
async def get_chat_session(session_id: str):
    logger.info(f"[Chat API] 세션 조회: {session_id}")
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    session = chat_sessions[session_id]
    return {
        "session_id": session_id,
        "messages": session.get("messages", []),
        "user_constraints": session.get("user_constraints", {})
    }