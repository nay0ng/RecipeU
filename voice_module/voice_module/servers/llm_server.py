"""
Qwen LLM FastAPI 서버 (venv_llm 환경 전용)

실행 방법:
    source venv_llm/bin/activate
    python servers/llm_server.py

    또는:
    uvicorn servers.llm_server:app --host 0.0.0.0 --port 8013

API 엔드포인트:
    POST /classify - 인텐트 분류 및 응답 생성
    GET /health - 서버 상태 확인
"""

import os
import sys
import re
import json
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv  # 추가: 환경변수 로드

import torch
import yaml
from fastapi import FastAPI, HTTPException, Security, Depends  # 추가: 보안 모듈
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader  # 추가: API Key 헤더
from starlette.status import HTTP_403_FORBIDDEN    # 추가: 403 상태코드
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================================
# 설정
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VLLM_ENDPOINT = "http://localhost:5000/v1"

PROMPTS_YAML_PATH = "./config/prompts.yaml"
USE_VLLM = True  # True: vLLM 사용, False: transformers 사용

# Intent 매핑
INTENT_MAP = {
    "Next": "next_step",
    "Prev": "prev_step",
    "Finish": "finish",
    "Missing Ingredient": "substitute_ingredient",
    "Missing Tool": "substitute_tool",
    "Failure": "failure",
    "Out of Scope": "unknown",
}

# ============================================================================
# 🔒 보안 설정 (API Key)
# ============================================================================

# 환경변수에서 키 가져오기 (RECIPEU_API_KEY 사용)
API_KEY = os.environ.get("RECIPEU_API_KEY")

# [안전장치] 키가 설정되지 않았으면 로그 경고
if not API_KEY:
    logger.error("❌ 치명적 오류: 'RECIPEU_API_KEY' 환경변수가 설정되지 않았습니다!")
    # 실전 배포 시 아래 주석 해제 권장
    # raise ValueError("RECIPEU_API_KEY 환경변수 미설정")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """API Key 검증 함수"""
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="인증 실패: 유효하지 않은 API Key입니다."
        )

# ============================================================================
# Pydantic 모델
# ============================================================================

class ClassifyRequest(BaseModel):
    """인텐트 분류 요청"""
    text: str = Field(..., description="사용자 발화", min_length=1)
    current_step: str = Field(default="", description="현재 요리 단계")
    current_cook: str = Field(default="", description="현재 요리 제목")
    recipe_context: str = Field(default="", description="인접 단계 정보")
    history: list = Field(default=[], description="대화 기록 [{role, content}, ...]")

class ClassifyResponse(BaseModel):
    """인텐트 분류 응답"""
    success: bool
    intent: str
    response: str
    raw_output: Optional[str] = None
    duration_ms: Optional[float] = None

class HealthResponse(BaseModel):
    """서버 상태 응답"""
    status: str
    model_loaded: bool
    model_name: str
    device: str
    uptime_seconds: float

# ============================================================================
# 전역 변수
# ============================================================================

llm_pipe: Optional[pipeline] = None
tokenizer: Optional[AutoTokenizer] = None

# vLLM 전용 변수
client: Optional[OpenAI] = None
prompts: Optional[dict] = None
server_start_time: float = 0

# ============================================================================
# 유틸리티 함수
# ============================================================================

def load_prompts():
    """프롬프트 YAML 로드"""
    global prompts
    logger.info(f"프롬프트 로드: {PROMPTS_YAML_PATH}")
    try:
        with open(PROMPTS_YAML_PATH, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        logger.info("프롬프트 로드 완료")
    except Exception as e:
        logger.error(f"프롬프트 로드 실패: {e}")
        raise

def get_prompt(key: str, **kwargs) -> str:
    """프롬프트 템플릿 가져오기"""
    template = prompts.get(key, {}).get('template', "")
    # 누락된 키는 빈 문자열로 처리
    return template.format_map(defaultdict(str, kwargs))

def extract_json(text: str) -> Dict[str, Any]:
    """LLM 출력에서 JSON 추출"""
    if not text:
        return {}

    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        logger.warning(f"JSON을 찾을 수 없음: {text[:100]}")
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}")
        return {}

# ============================================================================
# 서버 시작/종료 이벤트
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드"""
    global llm_pipe, tokenizer, client, server_start_time

    logger.info("=" * 60)
    logger.info("Qwen LLM 서버 시작")
    logger.info("=" * 60)
    logger.info(f"모델: {MODEL_NAME}")
    logger.info(f"디바이스: {DEVICE}")
    logger.info(f"vLLM 사용: {USE_VLLM}")

    # 프롬프트 로드
    load_prompts()

    logger.info("Qwen 모델 로딩 중... (시간이 걸릴 수 있습니다)")
    start_time = time.time()

    try:
        if USE_VLLM:
            client = OpenAI(base_url=VLLM_ENDPOINT, api_key="vllm")
            logger.info(f"vLLM 클라이언트 연결 설정 완료: {VLLM_ENDPOINT}")
        else:
            # Tokenizer 로드
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_fast=False
            )

            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # Pipeline 생성
            llm_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
            logger.info("transformers 모델 로드 완료")

        load_time = time.time() - start_time
        logger.info(f"모델 로딩 완료! (소요 시간: {load_time:.2f}초)")

        server_start_time = time.time()
        logger.info("서버 준비 완료!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise

    yield

    # 종료
    logger.info("서버 종료 중...")
    llm_pipe = None
    client = None
    server_start_time = 0
    tokenizer = None
    logger.info("서버 종료 완료")

# ============================================================================
# FastAPI 앱 생성
# ============================================================================

app = FastAPI(
    title="Qwen LLM API",
    description="Qwen 기반 인텐트 분류 및 응답 생성 API 서버",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(get_api_key)]  # ⭐ 모든 요청에 API Key 인증 적용
)

# CORS 설정 (보안 강화)
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://recipeu.site"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["시스템"])
async def health_check():
    """서버 상태 확인"""
    uptime = time.time() - server_start_time if server_start_time > 0 else 0

    # vLLM 또는 transformers 모델 로드 여부 확인
    model_loaded = client is not None if USE_VLLM else llm_pipe is not None

    return HealthResponse(
        status="healthy" if model_loaded else "loading",
        model_loaded=model_loaded,
        model_name=MODEL_NAME,
        device=DEVICE,
        uptime_seconds=round(uptime, 2),
    )

@app.post("/classify", response_model=ClassifyResponse, tags=["인텐트"])
async def classify(request: ClassifyRequest):
    """
    인텐트 분류 및 응답 생성

    Parameters:
        text: 사용자 발화
        current_step: 현재 요리 단계

    Returns:
        ClassifyResponse: 인텐트 및 응답
    """
    # 모델 로드 확인
    if USE_VLLM:
        if not client:
            raise HTTPException(status_code=503, detail="SDK 클라이언트 미설정")
    else:
        if not llm_pipe:
            raise HTTPException(status_code=503, detail="모델이 아직 로딩 중입니다")

    logger.info(f"분류 요청: '{request.text[:50]}...'")
    start_time = time.time()

    try:
        # 대화 기록을 텍스트로 포매팅 (프롬프트 템플릿의 {chat_history}에 삽입)
        if request.history:
            chat_history_lines = []
            for h in request.history:
                role_label = "사용자" if h.get("role") == "user" else "AI"
                chat_history_lines.append(f"- {role_label}: {h.get('content', '')}")
            chat_history_text = "\n".join(chat_history_lines)
        else:
            chat_history_text = "(이전 대화 없음)"

        prompt = get_prompt(
            "unified_handler",
            text=request.text,
            current_step=request.current_step,
            current_cook=request.current_cook,
            recipe_context=request.recipe_context,
            chat_history=chat_history_text
        )

        # LLM 호출
        messages = [
            {"role": "system", "content": "너는 사용자의 요리 과정을 돕는 똑똑한 쉐프 조수야."},
            {"role": "user", "content": prompt}
        ]

        if USE_VLLM:
            chat_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=256
            )
            # 🔍 여기! 딱 이 줄 추가
            # logger.info(f"message dump: {chat_response.choices[0].message}")

            msg = chat_response.choices[0].message
            raw_output = (msg.content or "").strip()

            if not raw_output:
                logger.warning("Empty content from vLLM")
                raw_output = "{}"
            
            # raw_output = chat_response.choices[0].message.content.strip()
        else:
            outputs = llm_pipe(
                messages,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id
            )

            # 결과 추출
            raw_output = outputs[0]['generated_text'][-1]['content'].strip()

        logger.debug(f"LLM 원본 출력: {raw_output}")

        # JSON 파싱
        data = extract_json(raw_output) or {}

        # -------- Intent 추출 (None/키변형 방어) --------
        raw_intent_val = (
            data.get("Intent")
            or data.get("intent")
            or data.get("INTENT")
            or "Out of Scope"
        )

        raw_intent = str(raw_intent_val).strip()
        intent = INTENT_MAP.get(raw_intent, "unknown")

        # -------- Response 추출 (null -> "" 방어) --------
        response_val = (
            data.get("Response")
            or data.get("response")
            or data.get("responseText")
            or data.get("ResponseText")
            or ""
        )

        response = str(response_val).strip()

        duration_ms = (time.time() - start_time) * 1000

        logger.info(f"분류 완료: {intent} / '{response[:50]}...' ({duration_ms:.0f}ms)")

        return ClassifyResponse(
            success=True,
            intent=intent,
            response=response,
            raw_output=raw_output,
            duration_ms=round(duration_ms, 2)
        )

    except Exception as e:
        logger.error(f"분류 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분류 중 오류 발생: {str(e)}")

# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
================================================================================
Qwen LLM API 서버

API 문서: http://localhost:8013/docs
상태 확인: http://localhost:8013/health

주의: venv_llm 환경에서 실행해야 합니다!
    source venv_llm/bin/activate
    python servers/llm_server.py
================================================================================
""")

    uvicorn.run(
        "llm_server:app",
        host="0.0.0.0",
        port=8013,
        reload=False,
        workers=1,
    )