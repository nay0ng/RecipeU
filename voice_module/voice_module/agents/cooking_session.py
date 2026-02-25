"""
요리 세션 관리 에이전트 (MSA 버전)

CookingSession: 요리 진행 상태 관리 및 음성 인터랙션 처리
- LLM 서버와 HTTP 통신 (8013번 포트)
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from core.api_client import STTClient, TTSClient, LLMClient
from core.types import Intent
from core.text_processor import preprocess_for_tts
from config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# CookingSession (MSA)
# ============================================================================

@dataclass
class CookingSession:
    """
    요리 세션 관리 클래스 (MSA 버전)

    Attributes:
        recipe_json: 레시피 JSON (다른 팀이 생성)
        step_index: 현재 단계 인덱스 (0부터 시작)
        history: 대화 히스토리 (UI 표시용)
    """

    recipe_json: Optional[Dict[str, Any]] = None
    step_index: int = 0
    history: List[Dict[str, str]] = field(default_factory=list)

    # 클라이언트들
    stt_client: Optional[STTClient] = None
    tts_client: Optional[TTSClient] = None
    llm_client: Optional[LLMClient] = None

    # TTS 카운터 (파일명 중복 방지)
    tts_counter: int = 0

    def __post_init__(self):
        """초기화 - 모든 클라이언트 생성"""
        if self.stt_client is None:
            logger.info("STT 클라이언트 초기화")
            self.stt_client = STTClient(
                base_url=settings.STT_SERVER_URL,
                timeout=settings.API_TIMEOUT
            )

        if self.tts_client is None:
            logger.info("TTS 클라이언트 초기화")
            self.tts_client = TTSClient(
                base_url=settings.TTS_SERVER_URL,
                timeout=settings.API_TIMEOUT,
                default_tone=settings.TTS_DEFAULT_TONE
            )

        if self.llm_client is None:
            logger.info("LLM 클라이언트 초기화 (8013번 포트 서버 사용)")
            self.llm_client = LLMClient(
                base_url=settings.LLM_SERVER_URL,
                timeout=settings.LLM_API_TIMEOUT
            )

        logger.info("CookingSession 초기화 완료")

    # ========================================================================
    # 레시피 설정
    # ========================================================================

    def set_recipe(self, recipe_json: Dict[str, Any]):
        """
        레시피 설정

        Args:
            recipe_json: 레시피 JSON 데이터
                형식: {"title": "...", "steps": [{"no": 1, "desc": "..."}]}
        """
        self.recipe_json = recipe_json
        self.step_index = 0
        self.history.clear()
        self.tts_counter = 0

        logger.info(f"레시피 설정: {recipe_json.get('title', 'Unknown')}")

    # ========================================================================
    # 음성 파일 처리 (E2E)
    # ========================================================================

    def handle_audio_file(self, audio_path: str) -> Tuple[str, str, int]:
        """
        음성 파일을 받아서 전체 파이프라인 실행
        STT -> LLM -> TTS

        Args:
            audio_path: 음성 파일 경로

        Returns:
            (응답 텍스트, TTS 음성 파일 경로, 현재 단계 인덱스)
        """
        logger.info(f"음성 파일 처리 시작: {audio_path}")
        start_time = time.time()

        try:
            # 1. STT
            stt_start = time.time()
            user_text = self.stt_client.transcribe(audio_path)
            stt_duration = time.time() - stt_start

            # 히스토리 기록
            if user_text.strip():
                self.history.append({"role": "user", "content": f"🎤 {user_text}"})
            else:
                self.history.append({"role": "user", "content": "🎤 [인식 실패]"})
                response = "음성을 인식하지 못했어요. 다시 한 번 또렷하게 말해 주세요."
                tts_path = self._generate_tts(response)
                self.history.append({"role": "assistant", "content": response})

                logger.warning("STT 인식 실패")
                return response, tts_path, self.step_index

            # 2. 텍스트 처리 (LLM 서버 호출)
            llm_start = time.time()
            response_text, step_idx = self.handle_text(user_text)
            llm_duration = time.time() - llm_start

            # 3. TTS
            tts_start = time.time()
            tts_path = self._generate_tts(response_text)
            tts_duration = time.time() - tts_start

            total_duration = time.time() - start_time

            logger.info(
                f"[LATENCY] STT={stt_duration:.2f}s | LLM={llm_duration:.2f}s | "
                f"TTS={tts_duration:.2f}s | Total={total_duration:.2f}s"
            )

            return response_text, tts_path, step_idx

        except Exception as e:
            logger.error(f"음성 파일 처리 실패: {e}")
            error_msg = f"처리 중 오류가 발생했어요: {str(e)}"
            self.history.append({"role": "assistant", "content": error_msg})
            tts_path = self._generate_tts(error_msg)
            return error_msg, tts_path, self.step_index

    # ========================================================================
    # 텍스트 처리 (챗봇 모드)
    # ========================================================================

    def handle_text(self, user_text: str) -> Tuple[str, int]:
        """
        사용자 텍스트를 직접 처리 (챗봇 모드)

        Args:
            user_text: 사용자 입력 텍스트

        Returns:
            (응답 텍스트, 현재 단계 인덱스)
        """
        if not self.recipe_json:
            return "레시피가 아직 설정되지 않았어요. 먼저 레시피를 생성해 주세요.", self.step_index

        logger.info(f"텍스트 처리: '{user_text[:50]}...'")

        try:
            # 현재 단계 정보
            current_step = self._get_current_step_desc()

            # LLM 서버 호출 (인텐트 분류 + 응답 생성)
            intent_str, llm_response = self.llm_client.classify_and_respond(
                user_text=user_text,
                current_step=current_step
            )

            # Intent 문자열을 Enum으로 변환
            intent = Intent(intent_str)
            logger.info(f"인텐트: {intent}")

            # 인텐트별 액션 실행
            response = self._process_intent(intent, llm_response)

            # 히스토리 기록
            self.history.append({"role": "assistant", "content": response})

            return response, self.step_index

        except Exception as e:
            logger.error(f"텍스트 처리 실패: {e}")
            error_msg = "요청을 처리할 수 없어요. 다시 시도해 주세요."
            self.history.append({"role": "assistant", "content": error_msg})
            return error_msg, self.step_index

    # ========================================================================
    # 인텐트 처리
    # ========================================================================

    def _process_intent(self, intent: Intent, llm_response: str) -> str:
        """
        인텐트별 액션 실행

        Args:
            intent: 분류된 인텐트
            llm_response: LLM이 생성한 응답

        Returns:
            최종 응답 텍스트
        """
        if intent == Intent.NEXT:
            return self._go_next()

        elif intent == Intent.PREV:
            return self._go_prev()

        elif intent in (Intent.SUB_ING, Intent.SUB_TOOL, Intent.FAILURE):
            # LLM 응답 그대로 사용
            return llm_response if llm_response else "처리할 수 없어요."

        else:  # Intent.UNKNOWN
            if llm_response:
                return llm_response
            else:
                return (
                    "지금은 조리 중이라 다음 기능만 지원해요: "
                    "다음/이전 단계, 재료 대체, 도구 대체, 실패 대응. "
                    "예) '다음', '이전', '재료가 없어서 대체 추천해줘', '냄비 없는데 어떻게 해?'"
                )

    # ========================================================================
    # 단계 이동
    # ========================================================================

    def _go_next(self) -> str:
        """다음 단계로 이동"""
        steps = self.recipe_json.get("steps", [])

        if self.step_index >= len(steps) - 1:
            return "마지막 단계예요. 완료되면 '요리 끝'이라고 말해 주세요."

        self.step_index += 1
        step = self._get_step(self.step_index)

        response = f"{step.get('no', self.step_index + 1)}단계: {step.get('desc', '')}"
        logger.info(f"다음 단계로 이동: {self.step_index}")

        return response

    def _go_prev(self) -> str:
        """이전 단계로 이동"""
        if self.step_index <= 0:
            step = self._get_step(0)
            return f"이미 1단계예요. 1단계: {step.get('desc', '')}"

        self.step_index -= 1
        step = self._get_step(self.step_index)

        response = f"{step.get('no', self.step_index + 1)}단계: {step.get('desc', '')}"
        logger.info(f"이전 단계로 이동: {self.step_index}")

        return response

    # ========================================================================
    # 유틸리티
    # ========================================================================

    def _get_step(self, index: int) -> Dict[str, Any]:
        """
        특정 인덱스의 단계 가져오기

        Args:
            index: 단계 인덱스

        Returns:
            단계 딕셔너리
        """
        steps = self.recipe_json.get("steps", [])
        if 0 <= index < len(steps):
            return steps[index]
        return {}

    def _get_current_step_desc(self) -> str:
        """현재 단계 설명 가져오기"""
        step = self._get_step(self.step_index)
        return step.get("desc", "")

    def _generate_tts(self, text: str) -> str:
        """
        TTS 음성 생성

        Args:
            text: 합성할 텍스트

        Returns:
            생성된 음성 파일 경로
        """
        # 텍스트 정제
        clean_text = preprocess_for_tts(text)

        # 출력 경로 생성
        os.makedirs(settings.TTS_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            settings.TTS_OUTPUT_DIR,
            f"tts_{self.tts_counter:04d}.wav"
        )

        self.tts_counter += 1

        # TTS 생성
        try:
            tts_path = self.tts_client.synthesize(
                text=clean_text,
                output_path=output_path,
                tone=settings.TTS_DEFAULT_TONE,
                speed_factor=settings.TTS_SPEED_FACTOR
            )
            return tts_path
        except Exception as e:
            logger.error(f"TTS 생성 실패: {e}")
            return ""

    # ========================================================================
    # 현재 단계 안내
    # ========================================================================

    def speak_current_step(self) -> str:
        """
        현재 단계 안내 음성 생성

        Returns:
            생성된 음성 파일 경로
        """
        if not self.recipe_json:
            message = "레시피가 설정되지 않았어요."
        else:
            step = self._get_step(self.step_index)
            title = self.recipe_json.get("title", "요리")
            message = f"지금부터 {title} 조리를 시작할게요. {step.get('no', 1)}단계 안내입니다. {step.get('desc', '')}"

        self.history.append({"role": "assistant", "content": message})
        tts_path = self._generate_tts(message)

        return tts_path

    # ========================================================================
    # 헬스체크
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        시스템 상태 확인

        Returns:
            상태 정보 딕셔너리
        """
        status = {
            "recipe_loaded": self.recipe_json is not None,
            "current_step": self.step_index,
            "total_steps": len(self.recipe_json.get("steps", [])) if self.recipe_json else 0,
            "history_count": len(self.history),
        }

        # STT 서버 확인
        try:
            stt_health = self.stt_client.health_check()
            status["stt_server"] = "healthy"
        except Exception as e:
            status["stt_server"] = f"error: {e}"

        # TTS 서버 확인
        try:
            tts_health = self.tts_client.health_check()
            status["tts_server"] = "healthy"
        except Exception as e:
            status["tts_server"] = f"error: {e}"

        # LLM 서버 확인
        try:
            llm_health = self.llm_client.health_check()
            status["llm_server"] = "healthy"
        except Exception as e:
            status["llm_server"] = f"error: {e}"

        return status
