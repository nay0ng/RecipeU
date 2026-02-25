"""
STT, TTS, LLM API 클라이언트

STTClient: Whisper 서버 (8011번 포트) 호출
TTSClient: GPT-SoVITS 서버 (8012번 포트) 호출
LLMClient: Qwen LLM 서버 (8013번 포트) 호출
"""

import os
import time
import logging
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ============================================================================
# STT Client
# ============================================================================

class STTClient:
    """Whisper STT 서버 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:8011", timeout: int = 30):
        """
        Args:
            base_url: STT 서버 URL
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def health_check(self) -> dict:
        """서버 상태 확인"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"STT 서버 연결 실패: {e}")
            raise RuntimeError(f"STT 서버에 연결할 수 없습니다: {e}")

    def transcribe(self, audio_path: str) -> str:
        """
        음성 파일을 텍스트로 변환

        Args:
            audio_path: 음성 파일 경로

        Returns:
            인식된 텍스트
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

        logger.info(f"STT 요청: {audio_path}")
        start_time = time.time()

        try:
            with open(audio_path, 'rb') as f:
                files = {'audio_file': (os.path.basename(audio_path), f)}
                response = requests.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    timeout=self.timeout
                )
                response.raise_for_status()

            result = response.json()
            text = result.get('text', '')
            duration = time.time() - start_time

            logger.info(f"STT 완료: '{text[:50]}...' ({duration:.2f}s)")
            return text

        except requests.exceptions.Timeout:
            raise RuntimeError(f"STT 요청 타임아웃 ({self.timeout}초)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"STT 요청 실패: {e}")
        except Exception as e:
            raise RuntimeError(f"STT 처리 중 오류: {e}")

# ============================================================================
# TTS Client
# ============================================================================

class TTSClient:
    """GPT-SoVITS TTS 서버 클라이언트"""

    def __init__(
        self,
        base_url: str = "http://localhost:8012",
        timeout: int = 30,
        default_tone: str = "kiwi"
    ):
        """
        Args:
            base_url: TTS 서버 URL
            timeout: 요청 타임아웃 (초)
            default_tone: 기본 톤 (레퍼런스 이름)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.default_tone = default_tone

    def health_check(self) -> dict:
        """서버 상태 확인"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"TTS 서버 연결 실패: {e}")
            raise RuntimeError(f"TTS 서버에 연결할 수 없습니다: {e}")

    def synthesize(
        self,
        text: str,
        output_path: str,
        tone: Optional[str] = None,
        text_lang: str = "ko",
        speed_factor: float = 1.0
    ) -> str:
        """
        텍스트를 음성으로 합성

        Args:
            text: 합성할 텍스트
            output_path: 저장할 파일 경로
            tone: 레퍼런스 톤 이름 (None이면 default_tone 사용)
            text_lang: 텍스트 언어 (ko, en, zh, ja)
            speed_factor: 음성 속도 (0.5~2.0)

        Returns:
            생성된 음성 파일 경로
        """
        if not text.strip():
            raise ValueError("텍스트가 비어있습니다")

        tone = tone or self.default_tone
        logger.info(f"TTS 요청: tone={tone}, text='{text[:50]}...'")
        start_time = time.time()

        try:
            # TTS 서버에 요청
            payload = {
                "text": text,
                "tone": tone,
                "text_lang": text_lang,
                "speed_factor": speed_factor,
                "save_file": True
            }

            response = requests.post(
                f"{self.base_url}/synthesize",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            if not result.get('success'):
                raise RuntimeError(f"TTS 실패: {result.get('message', 'Unknown error')}")

            # 서버에서 생성된 파일 경로
            server_audio_path = result.get('audio_path')

            if not server_audio_path or not os.path.exists(server_audio_path):
                raise RuntimeError(f"TTS 파일 생성 실패: {server_audio_path}")

            # 출력 경로로 복사
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            import shutil
            shutil.copy(server_audio_path, output_path)

            duration = time.time() - start_time
            logger.info(f"TTS 완료: {output_path} ({duration:.2f}s)")

            return output_path

        except requests.exceptions.Timeout:
            raise RuntimeError(f"TTS 요청 타임아웃 ({self.timeout}초)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"TTS 요청 실패: {e}")
        except Exception as e:
            raise RuntimeError(f"TTS 처리 중 오류: {e}")

    def list_references(self) -> list:
        """등록된 레퍼런스 목록 조회"""
        try:
            response = requests.get(
                f"{self.base_url}/references",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"레퍼런스 조회 실패: {e}")
            return []

# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """Qwen LLM 서버 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:8013", timeout: int = 60):
        """
        Args:
            base_url: LLM 서버 URL
            timeout: 요청 타임아웃 (초) - 첫 로딩 시 오래 걸릴 수 있음
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def health_check(self) -> dict:
        """서버 상태 확인"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LLM 서버 연결 실패: {e}")
            raise RuntimeError(f"LLM 서버에 연결할 수 없습니다: {e}")

    def classify_and_respond(
        self,
        user_text: str,
        current_step: str = ""
    ) -> Tuple[str, str]:
        """
        사용자 발화를 인텐트로 분류하고 응답 생성

        Args:
            user_text: 사용자 발화
            current_step: 현재 요리 단계

        Returns:
            (intent, response) 튜플
        """
        logger.info(f"LLM 요청: '{user_text[:50]}...'")
        start_time = time.time()

        try:
            payload = {
                "text": user_text,
                "current_step": current_step
            }

            response = requests.post(
                f"{self.base_url}/classify",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            if not result.get('success'):
                raise RuntimeError("LLM 분류 실패")

            intent = result.get('intent', 'unknown')
            response_text = result.get('response', '')

            duration = time.time() - start_time
            logger.info(f"LLM 완료: {intent} / '{response_text[:50]}...' ({duration:.2f}s)")

            return intent, response_text

        except requests.exceptions.Timeout:
            raise RuntimeError(f"LLM 요청 타임아웃 ({self.timeout}초)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM 요청 실패: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM 처리 중 오류: {e}")
