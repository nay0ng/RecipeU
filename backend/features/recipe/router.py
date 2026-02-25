# backend/features/recipe/router.py
"""
Recipe REST API 라우터
"""
import os
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pymongo import MongoClient

from core.dependencies import get_rag_system
from core.exceptions import RAGNotAvailableError
from features.recipe.service import RecipeService
from features.recipe.schemas import RecipeGenerateRequest
from .prompts import RECIPE_QUERY_EXTRACTION_PROMPT, RECIPE_GENERATION_PROMPT
from models.mysql_db import (
    save_my_recipe, get_my_recipes, get_my_recipe, delete_my_recipe, update_my_recipe,
    get_member_personalization, get_member_by_id,
    create_generate, get_generate, get_session_generates
)

router = APIRouter()


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

    # 1) 단계별 토큰 요약 표 (마크다운)
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

        # 2) 전체 합계 요약 표 (마크다운)
        print("\n- 📊 전체 합계 요약\n")
        print("| 구분 | Prompt Tokens | Completion Tokens | Total Tokens |")
        print("|------|---------------|-------------------|--------------|")
        print(f"| 합계 | {_token_accumulator['prompt']:,} | {_token_accumulator['completion']:,} | {_token_accumulator['total']:,} |")

    # 3) 성능 병목 표: 동작 플로우 순서대로 (마크다운)
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



def _format_elapsed_time(seconds) -> str:
    """초(int)를 HH:MM:SS 문자열로 변환"""
    if not seconds:
        return ""
    try:
        s = int(seconds)
        hrs = s // 3600
        mins = (s % 3600) // 60
        secs = s % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return ""


def get_user_profile_from_db(member_id: int) -> dict:
    """MySQL에서 사용자 프로필 조회"""
    if member_id == 0:
        return {"name": "게스트", "allergies": [], "dislikes": []}

    member = get_member_by_id(member_id)
    psnl = get_member_personalization(member_id)

    return {
        "name": member.get("nickname", "사용자") if member else "사용자",
        "allergies": psnl.get("allergies", []) if psnl else [],
        "dislikes": psnl.get("dislikes", []) if psnl else []
    }


@router.post("/generate")
async def generate_recipe(
    request: RecipeGenerateRequest,
    background_tasks: BackgroundTasks,
    rag_system = Depends(get_rag_system)
):
    """레시피 생성 (대화 히스토리 반영) - generate 테이블에 저장"""
    print("\n" + "="*60)
    print("[Recipe API] 레시피 생성 요청")
    print("="*60)

    if not rag_system:
        raise RAGNotAvailableError()

    # member_id 추출 (숫자면 사용, 아니면 0)
    member_id = 0
    if request.member_info:
        mid = request.member_info.get('member_id')
        if mid and str(mid).isdigit():
            member_id = int(mid)

    # MySQL에서 사용자 프로필 조회
    user_profile = get_user_profile_from_db(member_id)
    print(f"[Recipe API] 사용자 프로필: {user_profile}")

    service = RecipeService(rag_system, None, user_profile)

    try:
        recipe_data = await service.generate_recipe(
            chat_history=request.chat_history,
            member_info=request.member_info
        )

        generate_id = None
        # 백그라운드로 generate 테이블에 저장
        def save_to_generate():
            nonlocal generate_id
            try:
                # session_id가 없으면 None으로 저장 (직접 호출 시)
                session_id = request.member_info.get('session_id') if request.member_info else None
                if session_id and str(session_id).isdigit():
                    session_id = int(session_id)
                else:
                    session_id = None

                result = create_generate(
                    session_id=session_id,
                    member_id=member_id,
                    recipe_name=recipe_data.get('title', '추천 레시피'),
                    ingredients=recipe_data.get('ingredients', []),
                    steps=recipe_data.get('steps', []),
                    gen_type="FIRST"
                )
                generate_id = result.get('generate_id')
                print(f"[Recipe API] generate 테이블 저장 완료: ID={generate_id}")
            except Exception as e:
                print(f"[Recipe API] generate 저장 실패: {e}")

        background_tasks.add_task(save_to_generate)

        # 즉시 응답 (generate_id는 백그라운드 저장 후 결정됨)
        return {
            "recipe": recipe_data,
            "member_id": member_id,
            "title": recipe_data.get('title', '추천 레시피'),
            "constraints": request.member_info or {}
        }

    except Exception as e:
        print(f"[Recipe API] 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-from-chat")
async def generate_recipe_from_chat(
    session_id: str,
    background_tasks: BackgroundTasks,
    rag_system = Depends(get_rag_system)
):
    """채팅 세션에서 레시피 생성 → generate 테이블에 저장"""
    print("\n" + "="*60)
    print("[Recipe API] 채팅 세션에서 레시피 생성")
    print("="*60)

    from features.chat.router import chat_sessions

    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    if not rag_system:
        raise RAGNotAvailableError()

    session = chat_sessions[session_id]
    messages = session.get("messages", [])
    user_constraints = session.get("user_constraints", {})
    db_session_id = session.get("db_session_id")  # MySQL session.session_id
    print(f"db_session_id: {db_session_id}")

    # member_id 추출 (숫자면 사용, 아니면 0)
    member_id = session.get("member_id", 0)
    if not member_id and user_constraints:
        mid = user_constraints.get('member_id')
        if mid and str(mid).isdigit():
            member_id = int(mid)

    print(f"[Recipe API] ===== chat_sessions 상태 =====")
    print(f"[Recipe API] WS session_id: {session_id}")
    print(f"[Recipe API] member_id: {member_id} (type: {type(member_id).__name__})")
    print(f"[Recipe API] db_session_id: {db_session_id} (type: {type(db_session_id).__name__})")
    print(f"[Recipe API] user_constraints.member_id: {user_constraints.get('member_id')}")
    print(f"[Recipe API] =============================")

    # 세션에 저장된 user_profile 또는 MySQL에서 조회
    user_profile = session.get("user_profile") or get_user_profile_from_db(member_id)

    if not messages:
        raise HTTPException(status_code=400, detail="대화 내용이 없습니다")

    print(f"[Recipe API] 세션 메시지 수: {len(messages)}")
    print(f"[Recipe API] 사용자 프로필: {user_profile}")
    print(f"[Recipe API] DB session_id: {db_session_id}")

    service = RecipeService(rag_system, None, user_profile)

    try:
        # 대화 히스토리에서 마지막 레시피 찾기
        existing_recipe = None
        for msg in reversed(messages):
            if msg.get("role") in ("assistant", "AGENT"):
                content = msg.get("content", "")
                # 레시피 감지: "재료" + 이모지 마커
                if "재료" in content and ("⏱️" in content or "📊" in content):
                    existing_recipe = content
                    print(f"[Recipe API] 기존 레시피 발견 (길이: {len(content)} 자)")
                    print(f"[Recipe API] 레시피 일부: {content[:150]}...")
                    break

        if existing_recipe:
            # 기존 레시피로부터 상세 조리 과정 생성 (RAG 없이)
            print(f"[Recipe API] 기존 레시피 사용 → RAG 검색 생략")
            recipe_json = await service.generate_recipe_from_existing(
                recipe_content=existing_recipe,
                member_info=user_constraints
            )
        else:
            # 레시피가 없으면 RAG로 검색 후 생성
            print(f"[Recipe API] 기존 레시피 없음 → RAG 검색 진행")
            last_agent_msg = [m for m in messages if m.get("role") in ("assistant", "AGENT")]
            chat_for_recipe = last_agent_msg[-1:] if last_agent_msg else messages[-1:]
            recipe_json = await service.generate_recipe(
                chat_history=chat_for_recipe,
                member_info=user_constraints
            )

        print(f"[Recipe API] 레시피 생성 완료: {recipe_json.get('title')}")
        print(f"[Recipe API] 이미지: {recipe_json.get('image', 'None')[:60]}...")

        generate_id = None
        # generate 테이블에 저장 (동기로 저장하여 generate_id 반환)
        if not db_session_id:
            print(f"[Recipe API] ⚠️ db_session_id가 None - session 테이블에 세션이 생성되지 않았을 수 있습니다.")
        if member_id > 0:
            print(f"[Recipe API] generate 저장 시도 - member_id: {member_id}, db_session_id: {db_session_id}")
            try:
                # 해당 세션의 이전 생성 개수 확인
                existing = get_session_generates(db_session_id) if db_session_id else []
                print(f"이전 생성 개수: {existing}")
                gen_order = len(existing) + 1
                gen_type = "FIRST" if gen_order == 1 else "RETRY"

                result = create_generate(
                    session_id=db_session_id,
                    member_id=member_id,
                    recipe_name=recipe_json.get('title', '추천 레시피'),
                    ingredients=recipe_json.get('ingredients', []),
                    steps=recipe_json.get('steps', []),
                    gen_type=gen_type,
                    gen_order=gen_order
                )
                generate_id = result.get('generate_id')
                print(f"[Recipe API] ✅ generate 저장 완료: generate_id={generate_id}, db_session_id={db_session_id}, gen_order={gen_order}")
            except Exception as e:
                print(f"[Recipe API] ❌ generate 저장 실패: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[Recipe API] ⚠️ generate 저장 스킵 - member_id가 0입니다.")

        # 응답에 generate_id 포함 (마이레시피 저장 시 사용)
        return {
            "recipe": recipe_json,
            "member_id": member_id,
            "title": recipe_json.get("title"),
            "constraints": user_constraints,
            "session_id": session_id,
            "db_session_id": db_session_id,
            "generate_id": generate_id
        }

    except Exception as e:
        print(f"[Recipe API] 레시피 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_recipes(
    member_id: int = Query(default=0),
    limit: int = Query(default=50),
):
    """저장된 레시피 목록 조회"""
    try:
        rows = get_my_recipes(member_id, limit)
        recipes = []
        for row in rows:
            recipes.append({
                "id": row.get("my_recipe_id"),
                "title": row.get("recipe_name"),
                "created_at": row.get("created_at"),
                "image": row.get("image_url", ""),
                "rating": row.get("rating") or 0,
                "ingredients": row.get("ingredients", []),
                "steps": row.get("steps", []),
                "cook_time": row.get("cook_time", ""),
                "level": row.get("level", ""),
                "cooking_time": _format_elapsed_time(row.get("elapsed_time")),
            })
        return {"recipes": recipes}
    except Exception as e:
        print(f"[Recipe API] 목록 조회 실패: {e}")
        return {"recipes": []}


@router.get("/{recipe_id}")
async def get_recipe_detail(
    recipe_id: int,
):
    """레시피 상세 조회"""
    try:
        row = get_my_recipe(recipe_id)
        if not row:
            raise HTTPException(status_code=404, detail="레시피를 찾을 수 없습니다")

        return {
            "id": row.get("my_recipe_id"),
            "member_id": row.get("member_id"),
            "title": row.get("recipe_name"),
            "recipe": {
                "title": row.get("recipe_name"),
                "ingredients": row.get("ingredients", []),
                "steps": row.get("steps", []),
                "image": row.get("image_url", ""),
                "cook_time": row.get("cook_time", ""),
                "level": row.get("level", ""),
                "cooking_time": _format_elapsed_time(row.get("elapsed_time")),
            },
            "rating": row.get("rating") or 0,
            "created_at": row.get("created_at"),
            "cooking_time": _format_elapsed_time(row.get("elapsed_time")),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Recipe API] 상세 조회 실패: {e}")
        raise HTTPException(status_code=503, detail="DB 조회 실패")

@router.post("/save-my-recipe")
async def save_recipe_to_mypage(
    request: dict,
):
    """요리 완료 후 마이레시피에 저장 (generate_id, session_id 연결)"""
    # 게스트 계정 ID (마이레시피 저장 불가)
    GUEST_MEMBER_ID = 2

    try:
        print(f"[Recipe API] 마이레시피 저장 요청 수신: {request}")
        # user_id 추출 (프론트엔드는 user_id로 전송)
        user_id = request.get("user_id")

        # member_id로 변환
        member_id = int(user_id) if user_id and str(user_id).isdigit() else 0

        # 게스트(user_id=0, None, 또는 GUEST_MEMBER_ID=2)는 저장 불가
        if not user_id or member_id in [0, GUEST_MEMBER_ID]:
            raise HTTPException(
                status_code=400, 
                detail="로그인이 필요한 기능입니다. 게스트는 레시피를 저장할 수 없습니다."
            )
        
        # member_id로 변환
        member_id = int(user_id) if str(user_id).isdigit() else 0
        if member_id == 0:
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 사용자 ID입니다."
            )

        # generate_id, session_id 추출
        raw_generate_id = request.get("generate_id")
        raw_session_id = request.get("session_id") or request.get("db_session_id")
        print(f"[Recipe API] 수신된 generate_id: {raw_generate_id} (type: {type(raw_generate_id).__name__})")
        print(f"[Recipe API] 수신된 session_id: {raw_session_id} (type: {type(raw_session_id).__name__})")

        generate_id = raw_generate_id
        if generate_id is not None:
            try:
                generate_id = int(generate_id)
            except (ValueError, TypeError):
                generate_id = None

        session_id = raw_session_id
        if session_id is not None:
            try:
                session_id = int(session_id)
            except (ValueError, TypeError):
                session_id = None

        print(f"[Recipe API] 변환된 generate_id: {generate_id}, session_id: {session_id}")

        recipe = request.get("recipe", {})

        # name → title 변환 (프론트엔드 호환성)
        recipe_title = recipe.get("title") or recipe.get("name", "마이레시피")

        # 재료와 레시피 단계 추출
        ingredients = recipe.get("ingredients", [])
        steps = recipe.get("steps", [])

        print(f"[Recipe API] 저장할 레시피: {recipe_title}")
        print(f"[Recipe API] ingredients 개수: {len(ingredients)}, 내용: {ingredients[:2] if ingredients else '없음'}...")
        print(f"[Recipe API] steps 개수: {len(steps)}, 내용: {steps[:2] if steps else '없음'}...")

        if not ingredients:
            print(f"[Recipe API] 경고: ingredients가 비어있습니다!")
        if not steps:
            print(f"[Recipe API] 경고: steps가 비어있습니다!")

        # elapsed_time 추출 (초 단위)
        elapsed_time = request.get("elapsed_time")
        if elapsed_time is not None:
            try:
                elapsed_time = int(elapsed_time)
            except (ValueError, TypeError):
                elapsed_time = None

        result = save_my_recipe(
            member_id=member_id,
            recipe_name=recipe_title,
            ingredients=ingredients,
            steps=steps,
            rating=request.get("rating"),
            image_url=recipe.get("image", ""),
            session_id=session_id,
            generate_id=generate_id,
            cook_time=recipe.get("cook_time", ""),
            level=recipe.get("level", ""),
            elapsed_time=elapsed_time,
        )

        print(f"[Recipe API] 마이레시피 저장: ID={result.get('my_recipe_id')}, member_id={member_id}, generate_id={generate_id}")

        return {
            "success": True,
            "recipe_id": result.get("my_recipe_id"),
            "message": "마이레시피에 저장되었습니다"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Recipe API] 마이레시피 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.delete("/{recipe_id}")
async def delete_recipe(recipe_id: int):
    """마이레시피 삭제"""
    try:
        existing = get_my_recipe(recipe_id)
        if not existing:
            raise HTTPException(status_code=404, detail="레시피를 찾을 수 없습니다")

        delete_my_recipe(recipe_id)
        return {"success": True, "message": "레시피가 삭제되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Recipe API] 마이레시피 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{recipe_id}")
async def update_recipe(recipe_id: int, request: dict):
    """마이레시피 수정 (평점, 제목 등)"""
    try:
        existing = get_my_recipe(recipe_id)
        if not existing:
            raise HTTPException(status_code=404, detail="레시피를 찾을 수 없습니다")

        result = update_my_recipe(
            my_recipe_id=recipe_id,
            recipe_name=request.get("title"),
            rating=request.get("rating"),
            image_url=request.get("image")
        )

        return {
            "success": True,
            "recipe": {
                "id": result.get("my_recipe_id"),
                "title": result.get("recipe_name"),
                "rating": result.get("rating"),
                "image": result.get("image_url")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Recipe API] 마이레시피 수정 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))