"""
GPT-SoVITS TTS API 서버

실행 방법:
    uvicorn tts_server:app --host 0.0.0.0 --port 8012

    또는 직접 실행:
    python tts_server.py

API 엔드포인트:
    POST /synthesize - 텍스트를 음성으로 합성
    GET /references - 등록된 레퍼런스 목록 조회
    POST /references - 새 레퍼런스 등록
    GET /health - 서버 상태 확인
"""
# !!!!!!!!!stream 처리로 추가됨
import soundfile as sf
import numpy as np

import os
import io
import time
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, Depends # Security, Depends 추가
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader            # 추가
from starlette.status import HTTP_403_FORBIDDEN              # 추가
from pydantic import BaseModel, Field
from dotenv import load_dotenv # 추가

from tts_inference import TTSInference, OUTPUTS_DIR

from pathlib import Path

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 🔒 보안 설정 (API Key - 실전형)
# ============================================================================

# 1. 환경변수에서 비밀키 가져오기
API_KEY = os.environ.get("RECIPEU_API_KEY")

# [중요] 키가 설정되지 않았으면 서버를 켜지 않고 에러를 내는 것이 안전합니다.
if not API_KEY:
    # 런팟 로그 확인용 (필요시 주석 처리 가능)
    logger.error("❌ 치명적 오류: 'RECIPEU_API_KEY' 환경변수가 설정되지 않았습니다!")
    # raise ValueError("❌ 치명적 오류: 'RECIPEU_API_KEY' 환경변수가 설정되지 않았습니다!")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 2. 검증 함수
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="인증 실패: 유효하지 않은 API Key입니다."
        )


# ============================================================================
# Pydantic 모델 정의
# 데이터 유효성 검사(Data Validation)와 설정 관리(Settings Management)를 쉽게 해주는 강력한 라이브러리
# ============================================================================

class SynthesizeRequest(BaseModel):
    """음성 합성 요청"""
    text: str = Field(..., description="합성할 텍스트", min_length=1, max_length=500)
    tone: str = Field(default="kiwi", description="레퍼런스 톤 이름")
    text_lang: str = Field(default="ko", description="텍스트 언어 (ko, en, zh, ja)")
    speed_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="음성 속도 (0.5~2.0)")
    save_file: bool = Field(default=True, description="파일로 저장할지 여부")


class ReferenceRequest(BaseModel):
    """레퍼런스 등록 요청"""
    name: str = Field(..., description="레퍼런스 이름", min_length=1)
    audio_path: str = Field(..., description="오디오 파일 경로")
    text: str = Field(..., description="레퍼런스 오디오의 텍스트")
    lang: str = Field(default="ko", description="언어 코드")


class SynthesizeResponse(BaseModel):
    """음성 합성 응답"""
    success: bool
    message: str
    audio_path: Optional[str] = None
    duration_ms: Optional[float] = None


class ReferenceInfo(BaseModel):
    """레퍼런스 정보"""
    name: str
    audio_path: str
    text: str
    lang: str
    exists: bool


class HealthResponse(BaseModel):
    """서버 상태 응답"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    available_references: int


# ============================================================================
# 전역 변수
# ============================================================================

tts: Optional[TTSInference] = None
server_start_time: float = 0


# ============================================================================
# 서버 시작/종료 이벤트
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델 로드, 종료 시 정리"""
    global tts, server_start_time

    # 시작
    logger.info("=" * 60)
    logger.info("GPT-SoVITS TTS 서버 시작")
    logger.info("=" * 60)

    logger.info("모델 로딩 중... (시간이 걸릴 수 있습니다)")
    start_time = time.time()

    try:
        tts = TTSInference(device="cuda", is_half=True)
        load_time = time.time() - start_time
        logger.info(f"모델 로딩 완료! (소요 시간: {load_time:.2f}초)")

        # 기본 레퍼런스 등록 (tts_inference.py의 메서드 사용)
        registered = tts.register_default_references()
        logger.info(f"기본 레퍼런스 {registered}개 등록 완료")

        server_start_time = time.time()
        logger.info("서버 준비 완료!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise

    yield

    # 종료
    logger.info("서버 종료 중...")
    tts = None
    logger.info("서버 종료 완료")


# ============================================================================
# FastAPI 앱 생성
# ============================================================================

app = FastAPI(
    title="GPT-SoVITS TTS API",
    description="GPT-SoVITS 기반 음성 합성 API 서버",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(get_api_key)] # 모든 요청에 인증 적용 (추가됨)
)

# CORS 설정 (수정됨)
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
    ref_count = len(tts.TONE_REFERENCES) if tts else 0

    return HealthResponse(
        status="healthy" if tts else "loading",
        model_loaded=tts is not None,
        uptime_seconds=round(uptime, 2),
        available_references=ref_count,
    )


@app.get("/references", response_model=List[ReferenceInfo], tags=["레퍼런스"])
async def list_references():
    """등록된 레퍼런스 목록 조회"""
    if not tts:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중입니다")

    references = []
    for name, ref in tts.TONE_REFERENCES.items():
        references.append(ReferenceInfo(
            name=name,
            audio_path=ref.path,
            text=ref.text,
            lang=ref.lang,
            exists=os.path.exists(ref.path),
        ))

    return references


@app.post("/references", tags=["레퍼런스"])
async def register_reference(request: ReferenceRequest):
    """새 레퍼런스 등록"""
    if not tts:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중입니다")

    if not os.path.exists(request.audio_path):
        raise HTTPException(status_code=400, detail=f"오디오 파일이 존재하지 않습니다: {request.audio_path}")

    tts.set_reference(
        tone=request.name,
        audio_path=request.audio_path,
        text=request.text,
        lang=request.lang,
    )

    logger.info(f"레퍼런스 등록: {request.name}")

    return {"success": True, "message": f"레퍼런스 '{request.name}' 등록 완료"}

@app.post("/synthesize", tags=["합성"])
async def synthesize(request: SynthesizeRequest):
    if not tts:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중입니다")

    if request.tone not in tts.TONE_REFERENCES:
        available = list(tts.TONE_REFERENCES.keys())
        raise HTTPException(status_code=400, detail=f"존재하지 않는 레퍼런스: {request.tone}, 사용 가능: {available}")

    try:
        import soundfile as sf

        # numpy array로 합성 (이미 코드에 있는 함수)
        sample_rate, audio_data = tts.synthesize_to_bytes(
            text=request.text,
            tone=request.tone,
            text_lang=request.text_lang,
            speed_factor=request.speed_factor,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts.wav"}
        )

    except Exception as e:
        logger.error(f"합성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# !!!!!!!!!stream 처리로 추가됨
@app.post("/synthesize/stream", tags=["합성"])
async def synthesize_stream(request: SynthesizeRequest):
    """
    [속도 최적화] 스트리밍 응답 (TTFB 획기적 단축)
    """
    if not tts:
        raise HTTPException(status_code=503, detail="모델 로딩 중")

    # tone 검증 추가
    if request.tone not in tts.TONE_REFERENCES:
        available = list(tts.TONE_REFERENCES.keys())
        raise HTTPException(status_code=400, detail=f"존재하지 않는 레퍼런스: {request.tone}, 사용 가능: {available}")

    logger.info(f"⚡ 스트림 요청: tone={request.tone}, text={request.text[:20]}...")

    # 오디오 제너레이터 함수 정의
    def audio_generator():
        stream = tts.synthesize_stream_generator(
            text=request.text,
            tone=request.tone,
            text_lang=request.text_lang,
            speed_factor=request.speed_factor,
            top_k=5
        )

        for _sr, chunk in stream:
            # GPT-SoVITS가 이미 int16 (-32768 ~ 32767)을 반환
            yield chunk.tobytes()

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": "32000",
            "X-Channels": "1",
            "X-Sample-Width": "2",  # int16 = 2 bytes
            "X-Sample-Format": "int16",
            "Cache-Control": "no-cache"
        }
    )


@app.get("/audio/{filename}", tags=["파일"])
async def get_audio_file(filename: str):
    """생성된 오디오 파일 다운로드"""
    file_path = os.path.join(OUTPUTS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename,
    )


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
================================================================================
GPT-SoVITS TTS API 서버

API 문서: http://localhost:8012/docs
상태 확인: http://localhost:8012/health
================================================================================
""")

    uvicorn.run(
        "tts_server:app",
        host="0.0.0.0",
        port=8012,
        reload=False,
        workers=1,  # GPU 사용 시 단일 워커 권장
    )