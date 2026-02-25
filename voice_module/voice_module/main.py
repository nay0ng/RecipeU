"""
요리 세션 테스트 메인 코드

사용 방법:
1. STT 서버 시작: python servers/stt_server.py (포트 8011)
2. TTS 서버 시작: python servers/tts_server.py (포트 8012)
3. LLM 서버 시작: python servers/llm_server.py (포트 8013)
3. 메인 코드 실행: python main.py
"""

import os
import json
import logging

from agents.cooking_session import CookingSession

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# ============================================================================
# 샘플 레시피 로드
# ============================================================================

def load_sample_recipe() -> dict:
    """
    샘플 레시피 로드 (recipe_sample.jsonl 첫 줄)

    JSONL 형식:
        {"id": "레시피 이름", "step": ["단계1", "단계2", ...]}

    변환 후 형식:
        {"title": "레시피 이름", "steps": [{"no": 1, "desc": "단계1"}, ...]}
    """
    recipe_file = "./recipe_sample.jsonl"

    if not os.path.exists(recipe_file):
        logger.warning(f"레시피 파일이 없습니다: {recipe_file}")
        # 기본 레시피 반환
        return {
            "title": "김치찌개",
            "steps": [
                {"no": 1, "desc": "냄비에 물을 붓고 끓입니다."},
                {"no": 2, "desc": "김치와 돼지고기를 넣습니다."},
                {"no": 3, "desc": "양파와 두부를 넣고 5분간 끓입니다."},
                {"no": 4, "desc": "간을 맞추고 파를 올려 완성합니다."}
            ]
        }

    try:
        with open(recipe_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            raw_recipe = json.loads(first_line)

            # JSONL 형식 변환: {"id": "...", "step": [...]} → {"title": "...", "steps": [...]}
            title = raw_recipe.get("id", "Unknown")
            step_list = raw_recipe.get("step", [])

            # steps 배열 생성 (no와 desc 포함)
            steps = [
                {"no": idx + 1, "desc": step_desc}
                for idx, step_desc in enumerate(step_list)
            ]

            recipe = {
                "title": title,
                "steps": steps
            }

            logger.info(f"레시피 로드: {title} ({len(steps)}단계)")
            return recipe

    except Exception as e:
        logger.error(f"레시피 로드 실패: {e}")
        raise

# ============================================================================
# 예제 1: 텍스트 모드 (챗봇)
# ============================================================================

def test_text_mode():
    """텍스트 입력 모드 테스트"""
    logger.info("=" * 60)
    logger.info("예제 1: 텍스트 모드 (챗봇)")
    logger.info("=" * 60)

    # 세션 생성
    session = CookingSession()

    # 레시피 설정
    recipe = load_sample_recipe()
    session.set_recipe(recipe)

    # 시스템 헬스체크
    health = session.health_check()
    logger.info(f"시스템 상태: {health}")

    # 현재 단계 안내
    logger.info("\n[현재 단계 안내]")
    tts_path = session.speak_current_step()
    logger.info(f"TTS 파일: {tts_path}")

    # 텍스트 입력 테스트
    test_inputs = [
        "다음",
        "다음 단계로 넘어가줘",
        "이전",
        "양파가 없는데 대체할 수 있어?",
        "다음",
        "음식이 탔어 어떡해?",
        "다음"
    ]

    for user_input in test_inputs:
        logger.info(f"\n[사용자] {user_input}")
        response, step_idx = session.handle_text(user_input)
        logger.info(f"[어시스턴트] {response}")
        logger.info(f"현재 단계: {step_idx}")

    logger.info("\n" + "=" * 60)
    logger.info("텍스트 모드 테스트 완료")
    logger.info("=" * 60)

# ============================================================================
# 예제 2: 음성 모드 (E2E)
# ============================================================================

def test_audio_mode():
    """음성 파일 입력 모드 테스트"""
    logger.info("=" * 60)
    logger.info("예제 2: 음성 모드 (E2E)")
    logger.info("=" * 60)

    # 세션 생성
    session = CookingSession()

    # 레시피 설정
    recipe = load_sample_recipe()
    session.set_recipe(recipe)

    # 음성 파일 경로 (wavs 폴더의 샘플 파일들)
    audio_files = [
        "./wavs/next.wav",
        "./wavs/back.wav",
        "./wavs/next.wav",
    ]

    for audio_path in audio_files:
        if not os.path.exists(audio_path):
            logger.warning(f"음성 파일이 없습니다: {audio_path}")
            continue

        logger.info(f"\n[음성 파일] {audio_path}")

        try:
            response, tts_path, step_idx = session.handle_audio_file(audio_path)
            logger.info(f"[어시스턴트] {response}")
            logger.info(f"[TTS 파일] {tts_path}")
            logger.info(f"현재 단계: {step_idx}")
        except Exception as e:
            logger.error(f"음성 처리 실패: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("음성 모드 테스트 완료")
    logger.info("=" * 60)
    
def test_audio_mode_vad():
    """
    실시간 마이크 VAD 기반 음성 모드 테스트
    VAD(로컬 마이크) -> STT 서버 -> LLM(Qwen) -> TTS 서버
    """
    logger.info("=" * 60)
    logger.info("예제 4: 실시간 VAD 음성 모드 (Mic)")
    logger.info("=" * 60)

    from core.vad_audio import VADAudioRecorder  # (2)에서 옮긴 파일
    import threading
    from pathlib import Path

    session = CookingSession()
    recipe = load_sample_recipe()
    session.set_recipe(recipe)

    # 서버 상태 확인
    health = session.health_check()
    logger.info(f"시스템 상태: {health}")

    # VAD 출력 폴더 (입력 wav 저장)
    vad_dir = Path("./outputs/vad_inputs")
    vad_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    rec = VADAudioRecorder()

    try:
        logger.info("🎤 마이크 대기 시작. 말하면 자동으로 처리됩니다. (Ctrl+C로 종료)")
        for wav_path in rec.listen_and_record(stop_event=stop_event, out_dir=str(vad_dir)):
            if wav_path is None:
                logger.info("[VAD] timeout(None) 수신")
                continue

            logger.info(f"[VAD] 발화 파일 생성: {wav_path}")
            response, tts_path, step_idx = session.handle_audio_file(wav_path)

            logger.info(f"[어시스턴트] {response}")
            logger.info(f"[TTS 파일] {tts_path}")
            logger.info(f"현재 단계: {step_idx}")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: 종료")
    finally:
        stop_event.set()
        rec.close()


# ============================================================================
# 예제 3: 대화 히스토리 확인
# ============================================================================

def test_history():
    """대화 히스토리 확인"""
    logger.info("=" * 60)
    logger.info("예제 3: 대화 히스토리")
    logger.info("=" * 60)

    session = CookingSession()
    recipe = load_sample_recipe()
    session.set_recipe(recipe)

    # 몇 가지 대화 진행
    session.handle_text("다음")
    session.handle_text("양파 없어")
    session.handle_text("다음")

    # 히스토리 출력
    logger.info("\n[대화 히스토리]")
    for i, msg in enumerate(session.history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        logger.info(f"{i+1}. [{role}] {content[:80]}...")

    logger.info("\n" + "=" * 60)
    logger.info("히스토리 테스트 완료")
    logger.info("=" * 60)

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("""
================================================================================
요리 세션 음성 모듈 테스트

필수 사항:
1. STT 서버가 8011번 포트에서 실행 중이어야 합니다
   실행: python servers/stt_server.py

2. TTS 서버가 8012번 포트에서 실행 중이어야 합니다
   실행: python servers/tts_server.py

3. LLM 서버가 8013번 포트에서 실행 중이어야 합니다
   실행: python servers/llm_server.py
   (vLLM 서버 5000번 포트 필요)

3. 음성 파일 테스트 시 wavs 폴더에 샘플 파일 필요
================================================================================
""")

    # 테스트 선택
    print("\n테스트 선택:")
    print("1. 텍스트 모드 (챗봇)")
    print("2. 음성 모드 (E2E) - STT 서버 필요")
    print("3. 대화 히스토리")
    print("4. 전체 실행")

    choice = input("\n선택 (1-4): ").strip()

    if choice == "1":
        test_text_mode()
    elif choice == "2":
        test_audio_mode()
        test_audio_mode_vad()
    elif choice == "3":
        test_history()
    elif choice == "4":
        test_text_mode()
        test_history()
    else:
        logger.info("텍스트 모드만 실행합니다 (기본)")
        test_text_mode()

    logger.info("\n모든 테스트 완료!")

if __name__ == "__main__":
    main()
