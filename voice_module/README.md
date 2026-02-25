# Voice Module - 요리 세션 음성 파이프라인

STT(Whisper) → LLM(Qwen via vLLM) → TTS(GPT-SoVITS) 파이프라인을 통한 요리 세션 음성 인터랙션 시스템

## 프로젝트 구조

```
voice_module/
├── servers/                      # MSA 서버
│   ├── stt_server.py             # Whisper STT FastAPI 서버 (8011)
│   ├── llm_server.py             # Qwen LLM FastAPI 서버 (8013, vLLM 5000)
│   ├── tts_server.py             # GPT-SoVITS TTS FastAPI 서버 (8012)
│   └── tts_inference.py          # TTS 추론 로직
│
├── core/                         # 핵심 로직
│   ├── api_client.py             # STT/TTS/LLM API 클라이언트
│   ├── llm_engine.py             # Qwen LLM 인텐트 분류 엔진
│   ├── text_processor.py         # TTS용 텍스트 전처리
│   ├── types.py                  # Intent Enum 공통 타입
│   └── vad_audio.py              # Voice Activity Detection
│
├── agents/
│   └── cooking_session.py        # 요리 세션 관리 (핵심)
│
├── config/
│   ├── settings.py               # 전역 설정 (서버 URL, 모델, 타임아웃 등)
│   └── prompts.yaml              # LLM 프롬프트 템플릿
│
├── utils/
│   ├── audio_utils.py            # 오디오 파일 처리 유틸리티
│   ├── GPT-SoVITS/               # GPT-SoVITS 모델 (gitignore)
│   └── references/               # TTS 참조 음성 (gitignore)
│
├── main.py                       # 메인 실행 파일
├── recipe_sample.jsonl           # 샘플 레시피 데이터
├── 1_ready_vllm.sh               # vLLM 서버 준비 스크립트
├── 2_go_others.sh                # STT/TTS/LLM 서버 시작 스크립트
├── venv_llm_requirements.txt     # LLM 서버 의존성
├── test_tts_endpoints.py         # TTS 엔드포인트 테스트
├── test_vllm.py                  # vLLM 테스트
└── test_vllm_stream.py           # vLLM 스트리밍 테스트
```

## 아키텍처

```
┌──────────────────────────────────┐
│  Main Application (venv)         │
│  - CookingSession                │
└──────────────────────────────────┘
     ↓ HTTP     ↓ HTTP     ↓ HTTP
┌──────────┐ ┌──────────┐ ┌──────────┐
│STT Server│ │LLM Server│ │TTS Server│
│(Whisper) │ │ (Qwen)   │ │(GPT-So)  │
│  :8011   │ │  :8013   │ │  :8012   │
│  venv    │ │ venv_llm │ │  venv    │
└──────────┘ └──────────┘ └──────────┘
                  ↓
            ┌──────────┐
            │  vLLM    │
            │  :5000   │
            └──────────┘
```

## 인텐트 분류

| LLM 출력 (Intent) | 매핑 키 | 설명 | 예시 |
|-------------------|---------|------|------|
| Next | next_step | 다음 단계 이동 | "다음" |
| Prev | prev_step | 이전 단계 이동 | "이전" |
| Finish | finish | 요리 완료 | "다 됐어" |
| Missing Ingredient | substitute_ingredient | 재료 대체 요청 | "양파가 없는데 대체할 수 있어?" |
| Missing Tool | substitute_tool | 도구 대체 요청 | "냄비가 없어" |
| Failure | failure | 요리 실패 대응 | "음식이 탔어 어떡해?" |
| Out of Scope | unknown | 범위 밖 요청 | - |

## 실행 방법

### 1. 환경 준비

```bash
# STT/TTS/메인 앱용 venv
python -m venv venv
source venv/bin/activate
pip install torch torchaudio transformers fastapi uvicorn pydantic requests pyyaml

# LLM 서버용 venv_llm (transformers 버전 충돌 방지)
python -m venv venv_llm
source venv_llm/bin/activate
pip install -r venv_llm_requirements.txt
```

### 2. 서버 시작

```bash
# 터미널 0: vLLM 서버
vllm serve jjjunho/Qwen3-4B-Instruct-2507-Korean-AWQ --port 5000 --gpu-memory-utilization 0.6

# 터미널 1: LLM 서버 (venv_llm 환경)
source venv_llm/bin/activate
python servers/llm_server.py

# 터미널 2: STT 서버 (venv 환경)
source venv/bin/activate
python servers/stt_server.py

# 터미널 3: TTS 서버 (venv 환경)
source venv/bin/activate
python servers/tts_server.py
```

또는 스크립트 사용:
```bash
bash 1_ready_vllm.sh   # vLLM 준비
bash 2_go_others.sh    # STT/TTS/LLM 서버 시작
```

### 3. 메인 실행

```bash
source venv/bin/activate
python main.py
```

### 4. 헬스체크

```bash
curl http://localhost:8011/health  # STT
curl http://localhost:8012/health  # TTS
curl http://localhost:8013/health  # LLM
```

## 사용 예제

### 텍스트 모드

```python
from agents.cooking_session import CookingSession

session = CookingSession()
session.set_recipe(recipe)

response, step = session.handle_text("다음")
print(f"응답: {response}")
print(f"현재 단계: {step}")
```

### 음성 모드 (E2E)

```python
response, tts_path, step = session.handle_audio_file("user_voice.wav")
print(f"응답: {response}")
print(f"음성 파일: {tts_path}")
```

## 설정

`config/settings.py`에서 변경 가능:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| STT_SERVER_URL | localhost:8011 | STT 서버 |
| TTS_SERVER_URL | localhost:8012 | TTS 서버 |
| LLM_SERVER_URL | localhost:8013 | LLM 서버 |
| LLM_MODEL_NAME | jjjunho/Qwen3-4B-Instruct-2507-Korean-AWQ | LLM 모델 |
| TTS_DEFAULT_TONE | kiwi | TTS 레퍼런스 톤 |
| API_TIMEOUT | 30초 | STT/TTS 타임아웃 |
| LLM_API_TIMEOUT | 60초 | LLM 타임아웃 |

## 성능

| 항목 | 시간 |
|------|------|
| STT (Whisper) | ~2-4초 |
| LLM (Qwen 4B) | ~0.7초 |
| TTS (GPT-SoVITS) | ~1-3초 |
| E2E 전체 | ~4-9초 |

### 리소스 요구사항
- GPU 메모리: ~8-12GB (Whisper + Qwen + TTS 동시)
- CPU: 4코어 이상
- RAM: 16GB 이상

## API 문서

각 서버 시작 후 Swagger UI 접근 가능:

| 서버 | Swagger UI | 주요 엔드포인트 |
|------|-----------|----------------|
| STT | localhost:8011/docs | `POST /transcribe` |
| TTS | localhost:8012/docs | `POST /synthesize` |
| LLM | localhost:8013/docs | `POST /classify` |

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| STT/TTS/LLM 서버 연결 실패 | `curl http://localhost:{port}/health`로 확인 후 재시작 |
| CUDA out of memory | `nvidia-smi`로 GPU 확인, `settings.py`에서 `DEVICE = "cpu"` 변경 |
| 포트 충돌 | `lsof -i :{port}`로 확인 후 `kill -9 <PID>` |
| transformers 버전 충돌 | STT/TTS는 venv, LLM은 venv_llm 분리 사용 |
