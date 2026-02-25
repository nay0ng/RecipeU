# MSA 구조 실행 가이드

## 🏗️ 아키텍처

```
┌─────────────────────────────────────┐
│  Main Application (venv)            │
│  - CookingSession                   │
│  - port: N/A                        │
└─────────────────────────────────────┘
     ↓ HTTP    ↓ HTTP    ↓ HTTP
┌──────────┐ ┌──────────┐ ┌──────────┐
│STT Server│ │LLM Server│ │TTS Server│
│(Whisper) │ │ (Qwen)   │ │(GPT-So)  │
│  8011    │ │  8013    │ │  8012    │
│  venv    │ │venv_llm  │ │  venv    │
└──────────┘ └──────────┘ └──────────┘
```

---

## 📦 설치

### 1. venv 환경 (STT + TTS + 메인 앱)

```bash
cd voice_module

# 이미 있으면 재사용
source venv/bin/activate

# 필요시 패키지 확인
pip list | grep -E "torch|torchaudio|transformers|fastapi"
```

### 2. venv_llm 환경 (LLM 서버 전용) ⭐ 신규

```bash
cd voice_module

# 가상환경 생성
python -m venv venv_llm
source venv_llm/bin/activate

# 패키지 설치
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/peft.git
pip install fastapi uvicorn[standard] accelerate tiktoken einops
pip install pydantic pyyaml

# 또는 requirements 사용
pip install -r venv_llm_requirements.txt

# Qwen 모델 미리 다운로드 (선택적, 권장)
python << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-4B-Instruct-2507"
print("Tokenizer 다운로드...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
print("모델 다운로드 (~8GB)...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
print("완료!")
EOF
```

---

## 🚀 실행 순서

### ✅ 필수: 3개 서버 모두 실행 필요

### 터미널 0: vLLM 서버 (5000번) ⭐ 필수

```bash
vllm serve jjjunho/Qwen3-4B-Instruct-2507-Korean-AWQ --port 5000 --gpu-memory-utilization 0.6
```

### 터미널 1: LLM 서버 (8013번, venv_llm) ⭐

```bash
cd voice_module
source venv_llm/bin/activate

python servers/llm_server.py
```

**출력 예시:**
```
[INFO] Qwen LLM 서버 시작
[INFO] 모델: Qwen/Qwen3-4B-Instruct-2507
[INFO] 디바이스: cuda
[INFO] Qwen 모델 로딩 중...
[INFO] 모델 로딩 완료! (소요 시간: 45.23초)
[INFO] 서버 준비 완료!
INFO:     Uvicorn running on http://0.0.0.0:8013
```

**확인:**
```bash
curl http://localhost:8013/health
# {"status":"healthy","model_loaded":true, ...}
```

---

### 터미널 2: STT 서버 (8011번, venv)

```bash
cd voice_module
source venv/bin/activate

python servers/stt_server.py
```

**출력 예시:**
```
[INFO] Whisper STT 서버 시작
[INFO] Whisper 모델 로딩 중...
[INFO] Whisper 모델 로딩 완료! (소요 시간: 15.32초)
INFO:     Uvicorn running on http://0.0.0.0:8011
```

**확인:**
```bash
curl http://localhost:8011/health
# {"status":"healthy","model_loaded":true, ...}
```

---

### 터미널 3: TTS 서버 (8012번, venv)

```bash
cd voice_module
source venv/bin/activate

python servers/tts_server.py
```

**확인:**
```bash
curl http://localhost:8012/health
# {"status":"healthy","model_loaded":true, ...}
```

---

### 터미널 4: 메인 애플리케이션 (venv)

**3개 서버가 모두 실행 중인지 확인 후:**

```bash
cd voice_module
source venv/bin/activate

python main.py
```

**출력 예시:**
```
[INFO] STT 클라이언트 초기화
[INFO] TTS 클라이언트 초기화
[INFO] LLM 클라이언트 초기화 (8013번 포트 서버 사용)
[INFO] CookingSession 초기화 완료

테스트 선택:
1. 텍스트 모드 (챗봇)
...
```

---

## 🧪 테스트

### 1. 서버 헬스체크

```bash
# 모든 서버 확인
curl http://localhost:8011/health  # STT
curl http://localhost:8013/health  # LLM ⭐
curl http://localhost:8012/health  # TTS
```

### 2. LLM 서버 직접 테스트

```bash
curl -X POST http://localhost:8013/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "다음 단계로 넘어가줘", "current_step": "양파를 볶아주세요"}'

# 예상 응답:
# {
#   "success": true,
#   "intent": "next_step",
#   "response": "",
#   "duration_ms": 234.56
# }
```

### 3. 전체 파이프라인 테스트

```bash
python main.py
# 1번 선택 (텍스트 모드)
# 입력: "다음"
# 출력: "2단계: ..."
```

---

## 📊 성능 벤치마크

| 작업 | 소요 시간 |
|------|----------|
| LLM 서버 시작 (첫 실행) | 60초 (모델 다운로드 포함) |
| LLM 서버 시작 (캐시 사용) | 30초 |
| STT 서버 시작 | 15초 |
| TTS 서버 시작 | 20초 |
| **총 서버 시작 시간** | **65초** (병렬 실행 시) |
| | |
| `session = CookingSession()` | **0.5초** ⚡ |
| `handle_text("다음")` | 1.2초 (LLM API 호출 포함) |
| E2E (음성 → 응답) | 5초 |

---

## 🔧 트러블슈팅

### 문제 1: LLM 서버 연결 실패
```
RuntimeError: LLM 서버에 연결할 수 없습니다
```

**해결:**
```bash
# LLM 서버 실행 확인
curl http://localhost:8013/health

# 안 되면 재시작 (venv_llm 환경!)
source venv_llm/bin/activate
python servers/llm_server.py
```

---

### 문제 2: venv_llm에서 transformers 버전 확인
```bash
source venv_llm/bin/activate
pip show transformers

# 출력: Version: 5.0.1.dev0 (또는 최신)
```

---

### 문제 3: GPU 메모리 부족
```
CUDA out of memory
```

**해결:**
```bash
# 다른 프로세스 확인
nvidia-smi

# Qwen 모델만 실행 (STT/TTS 종료)
# 또는 CPU 모드로 변경 (servers/llm_server.py)
# DEVICE = "cpu"
```

---

### 문제 4: 포트 충돌
```
Address already in use: 8013
```

**해결:**
```bash
# 프로세스 확인
lsof -i :8013

# 종료
kill -9 <PID>

# 또는 다른 포트 사용
# config/settings.py: LLM_SERVER_URL = "http://localhost:8004"
# servers/llm_server.py: port=8004
```

---

## 💡 장점

### ✅ 의존성 충돌 해결
- venv: transformers 4.43~4.50 (GPT-SoVITS)
- venv_llm: transformers 최신 (Qwen)
- 완전 분리!

### ✅ 독립적 배포
- LLM 서버만 재시작 가능
- 메인 앱 중단 없음

### ✅ 빠른 시작
- `CookingSession()` 생성: 0.5초
- LLM 로딩은 서버에서 미리 완료

### ✅ 스케일링
- LLM 서버를 더 큰 GPU 인스턴스로 분리
- STT/TTS는 CPU 인스턴스

---

## 📁 변경된 파일

### 신규 파일
- `servers/llm_server.py` - Qwen LLM FastAPI 서버 (venv_llm)
- `core/types.py` - Intent Enum 공통 타입
- `venv_llm_requirements.txt` - LLM 서버 의존성
- `MSA_GUIDE.md` - 이 파일

### 수정된 파일
- `core/api_client.py` - LLMClient 추가
- `config/settings.py` - LLM_SERVER_URL 추가
- `agents/cooking_session.py` - LLM 서버 사용으로 변경

### 백업 파일
- `agents/cooking_session_lazy_backup.py` - 이전 버전 (Lazy Loading)
- `agents/cooking_session_eager.py.backup` - 더 이전 버전 (Eager Loading)

---

## 🎯 다음 단계

### Production 배포 시
1. **Docker Compose 사용**
   ```yaml
   services:
     llm:
       build: ./llm_server
       ports: ["8013:8013"]
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
     stt:
       build: ./stt_server
       ports: ["8011:8011"]
     tts:
       ...
   ```

2. **로드 밸런서**
   - Nginx로 LLM 서버 여러 개 로드 밸런싱

3. **모니터링**
   - Prometheus + Grafana
   - 각 서버별 메트릭 수집

---

## 📞 문의

문제 발생 시:
1. 각 서버 로그 확인
2. health check 엔드포인트 확인
3. GPU 메모리 확인 (`nvidia-smi`)
