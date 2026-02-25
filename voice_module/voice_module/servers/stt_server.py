"""
Whisper 기반 STT FastAPI 서버

실행 방법:
    cd /data/ephemeral/home/hyejin/260128_voice_module
    python servers/stt_server.py

    또는:
    uvicorn servers.stt_server:app --host 0.0.0.0 --port 8011

API 엔드포인트:
    POST /transcribe - 음성 파일을 텍스트로 변환
    GET /health - 서버 상태 확인
"""

import os
import io
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

from fastapi import Request
import numpy as np
import subprocess
import tempfile

# ============================================================================
# [추가] RunPod 환경에서 FFmpeg 경로 강제 등록
# ============================================================================
if os.path.exists("/workspace/bin/ffmpeg"):
    os.environ["PATH"] += os.pathsep + "/workspace/bin"
    print(f"FFmpeg 경로 추가됨: {os.environ['PATH']}")
# ============================================================================


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic 모델 정의
# ============================================================================

class TranscribeResponse(BaseModel):
    """STT 응답"""
    success: bool
    text: str
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

whisper_pipe: Optional[pipeline] = None
server_start_time: float = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/whisper-large-v3-turbo"

# ============================================================================
# 오디오 전처리 함수
# ============================================================================

def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    """
    오디오 바이트를 Whisper 입력 형식으로 전처리
    - 16kHz 모노 변환
    """
    # BytesIO로 변환
    audio_buffer = io.BytesIO(audio_bytes)

    # 오디오 로드
    waveform, sample_rate = torchaudio.load(audio_buffer)

    # 스테레오 → 모노
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 리샘플링 (16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # numpy array로 변환
    audio_array = waveform.squeeze().numpy()

    return audio_array


def decode_any_to_float32(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    다양한 포맷(webm/mp4/m4a/ogg/wav 등) bytes를 float32 mono 16k로 디코딩
    ffmpeg 필요
    """
    # ffmpeg로 raw PCM(float32) 뽑기
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in.flush()

        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", tmp_in.name,
            "-ac", "1",          # mono
            "-ar", "16000",      # 16k
            "-f", "f32le",       # float32 little endian raw
            "pipe:1",
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0 or not p.stdout:
            raise RuntimeError(f"ffmpeg decode failed: {p.stderr.decode('utf-8', errors='ignore')[:300]}")

        audio = np.frombuffer(p.stdout, dtype=np.float32)
        return audio, 16000

# ============================================================================
# 서버 시작/종료 이벤트
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 Whisper 모델 로드"""
    global whisper_pipe, server_start_time

    logger.info("=" * 60)
    logger.info("Whisper STT 서버 시작")
    logger.info("=" * 60)
    logger.info(f"모델: {MODEL_NAME}")
    logger.info(f"디바이스: {DEVICE}")

    logger.info("Whisper 모델 로딩 중... (시간이 걸릴 수 있습니다)")
    start_time = time.time()

    try:
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            chunk_length_s=30,
            stride_length_s=5,
            device=DEVICE
        )

        load_time = time.time() - start_time
        logger.info(f"Whisper 모델 로딩 완료! (소요 시간: {load_time:.2f}초)")

        server_start_time = time.time()
        logger.info("서버 준비 완료!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise

    yield

    # 종료
    logger.info("서버 종료 중...")
    whisper_pipe = None
    logger.info("서버 종료 완료")

# ============================================================================
# FastAPI 앱 생성
# ============================================================================

app = FastAPI(
    title="Whisper STT API",
    description="Whisper 기반 음성 인식 API 서버",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API 엔드포인트
# ============================================================================


def transcribe_from_bytes(audio_bytes: bytes) -> tuple[str, float]:
    if not whisper_pipe:
        raise RuntimeError("whisper_pipe not loaded")

    # ✅ 어떤 포맷이든 디코딩
    data, sr = decode_any_to_float32(audio_bytes)

    start = time.time()
    result = whisper_pipe(
        {"array": data, "sampling_rate": sr},
        generate_kwargs={"language": "ko", "task": "transcribe"},
        return_timestamps=False
    )
    dur_ms = (time.time() - start) * 1000
    text = result["text"] if isinstance(result, dict) and "text" in result else str(result)
    return text.strip(), round(dur_ms, 2)


@app.post("/transcribe_bytes")
async def transcribe_bytes(request: Request):
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty body")

    text, duration_ms = transcribe_from_bytes(audio_bytes)
    return {"success": True, "text": text, "duration_ms": duration_ms}



@app.get("/health", response_model=HealthResponse, tags=["시스템"])
async def health_check():
    """서버 상태 확인"""
    uptime = time.time() - server_start_time if server_start_time > 0 else 0

    return HealthResponse(
        status="healthy" if whisper_pipe else "loading",
        model_loaded=whisper_pipe is not None,
        model_name=MODEL_NAME,
        device=DEVICE,
        uptime_seconds=round(uptime, 2),
    )

@app.post("/transcribe", response_model=TranscribeResponse, tags=["STT"])
async def transcribe(audio_file: UploadFile = File(...)):
    """
    음성 파일을 텍스트로 변환 (STT)

    Parameters:
        audio_file: 음성 파일 (.wav, .m4a, .mp3 등)

    Returns:
        TranscribeResponse: 인식된 텍스트 및 메타데이터
    """
    if not whisper_pipe:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩 중입니다")

    # 파일 확장자 확인
    filename = audio_file.filename or "unknown"
    logger.info(f"STT 요청: {filename}")

    try:
        start_time = time.time()

        # 파일 읽기
        audio_bytes = await audio_file.read()

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다")

        # 오디오 전처리
        audio_array = preprocess_audio(audio_bytes)

        # Whisper 추론
        result = whisper_pipe(
            audio_array,
            generate_kwargs={
                "language": "ko",
                "task": "transcribe"
            }
        )

        text = result.get("text", "").strip()
        duration_ms = (time.time() - start_time) * 1000

        logger.info(f"STT 완료: '{text[:50]}...' ({duration_ms:.0f}ms)")

        return TranscribeResponse(
            success=True,
            text=text,
            duration_ms=round(duration_ms, 2)
        )

    except Exception as e:
        logger.error(f"STT 실패: {e}")
        raise HTTPException(status_code=500, detail=f"STT 처리 중 오류 발생: {str(e)}")

# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
================================================================================
Whisper STT API 서버

API 문서: http://localhost:8011/docs
상태 확인: http://localhost:8011/health
================================================================================
""")

    uvicorn.run(
        "stt_server:app",
        host="0.0.0.0",
        port=8011,
        reload=False,
        workers=1,
    )
