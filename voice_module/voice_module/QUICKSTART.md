# 빠른 시작 가이드 (Quick Start)

## 🎯 3단계로 시작하기

### 1단계: STT 서버 시작 (터미널 1)

```bash
python servers/stt_server.py
```

**성공 시 출력:**
```
[INFO] Whisper 모델 로딩 완료! (소요 시간: 15.23초)
[INFO] 서버 준비 완료!
INFO:     Uvicorn running on http://0.0.0.0:8011
```

### 2단계: TTS 서버 시작 (터미널 2)

```bash
python servers/tts_server.py
```

**확인:**
```bash
curl http://localhost:8012/health
```

### 3단계: vLLM + LLM 서버 시작 (터미널 3, 4)

vLLM 서버:
```bash
vllm serve jjjunho/Qwen3-4B-Instruct-2507-Korean-AWQ --port 5000 --gpu-memory-utilization 0.4
```

LLM 서버:
```bash
python servers/llm_server.py
```

### 4단계: 테스트 실행 (터미널 5)

```bash
python main.py
```

**메뉴 선택:**
```
테스트 선택:
1. 텍스트 모드 (챗봇)
2. 음성 모드 (E2E) - STT 서버 필요
3. 대화 히스토리
4. 전체 실행

선택 (1-4): 1
```

## ✅ 동작 확인

### 텍스트 모드 테스트
```python
# 예상 출력:
[INFO] LLM 모델 로드 완료
[INFO] 레시피 설정: 김치찌개
[INFO] 시스템 상태: {'recipe_loaded': True, 'stt_server': 'healthy', ...}

[사용자] 다음
[어시스턴트] 2단계: 김치와 돼지고기를 넣습니다.
현재 단계: 1

[사용자] 양파가 없는데 대체할 수 있어?
[어시스턴트] 양파 대신 대파나 쪽파를 사용할 수 있어요...
```

## 🔍 주요 파일 위치

| 파일 | 역할 |
|------|------|
| [servers/stt_server.py](servers/stt_server.py) | Whisper STT 서버 (8011) |
| [servers/tts_server.py](servers/tts_server.py) | GPT-SoVITS TTS 서버 (8012) |
| [servers/llm_server.py](servers/llm_server.py) | Qwen LLM 서버 (8013) |
| [agents/cooking_session.py](agents/cooking_session.py) | 요리 세션 관리 (핵심) |
| [core/llm_engine.py](core/llm_engine.py) | Qwen 인텐트 분류 |
| [core/api_client.py](core/api_client.py) | STT/TTS API 호출 |
| [config/prompts.yaml](config/prompts.yaml) | LLM 프롬프트 템플릿 |
| [main.py](main.py) | 테스트 실행 코드 |

## 🐛 문제 해결

### 문제 1: "STT 서버에 연결할 수 없습니다"
```bash
# 확인
curl http://localhost:8011/health

# 재시작
pkill -f stt_server
python servers/stt_server.py
```

### 문제 2: "CUDA out of memory"
```python
# config/settings.py 수정
DEVICE = "cpu"  # GPU → CPU로 변경
```

### 문제 3: LLM 로딩이 너무 느림
```
[INFO] LLM 모델 로드 시작: jjjunho/Qwen3-4B-Instruct-2507-Korean-AWQ
# 30초 ~ 1분 정도 대기 (정상)
```

## 📞 다음 단계

1. **음성 파일 테스트**: `wavs/` 폴더에 음성 파일 추가 후 `python main.py` 선택 2
2. **프롬프트 수정**: `config/prompts.yaml` 편집
3. **레시피 변경**: `recipe_sample.jsonl` 수정
4. **커스텀 통합**: `agents/cooking_session.py`를 자신의 앱에 임포트

## 📚 상세 문서

- [README.md](README.md) - 전체 문서
- [config/prompts.yaml](config/prompts.yaml) - 프롬프트 템플릿
- http://localhost:8011/docs - STT API 문서
- http://localhost:8012/docs - TTS API 문서
- http://localhost:8013/docs - LLM API 문서
