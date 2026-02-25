"""
Qwen LLM 엔진 - 인텐트 분류 및 응답 생성

사용법:
    engine = LLMEngine()
    intent, response = engine.classify_and_respond(
        user_text="다음 단계로 넘겨줘",
        current_step="양파를 볶아주세요"
    )
"""

import re
import json
import logging
from typing import Tuple, Optional, Dict, Any
from enum import Enum

from vllm import LLM, SamplingParams

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# ============================================================================
# Intent Enum
# ============================================================================

class Intent(str, Enum):
    """요리 세션 인텐트"""
    NEXT = "next_step"
    PREV = "prev_step"
    SUB_ING = "substitute_ingredient"
    SUB_TOOL = "substitute_tool"
    FAILURE = "failure"
    UNKNOWN = "unknown"

# ============================================================================
# LLM Engine
# ============================================================================

class LLMEngine:
    """Qwen 기반 인텐트 분류 및 응답 생성 엔진"""

    # Intent 매핑 (LLM 출력 → Intent Enum)
    INTENT_MAP = {
        "Next": Intent.NEXT,
        "Prev": Intent.PREV,
        "Missing Ingredient": Intent.SUB_ING,
        "Missing Tool": Intent.SUB_TOOL,
        "Failure": Intent.FAILURE,
        "Out of Scope": Intent.UNKNOWN,
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        prompts_yaml_path: str = "./config/prompts.yaml"
    ):
        """
        Args:
            model_name: Qwen 모델 이름
            device: 디바이스 (cuda/cpu)
            prompts_yaml_path: 프롬프트 YAML 파일 경로
        """
        self.model_name = model_name
        self.device = device
        self.prompts_yaml_path = prompts_yaml_path

        # 프롬프트 로드
        self._load_prompts()

        # 모델 로드
        self._load_model()

    def _load_prompts(self):
        """프롬프트 YAML 파일 로드"""
        logger.info(f"프롬프트 로드: {self.prompts_yaml_path}")
        try:
            with open(self.prompts_yaml_path, 'r', encoding='utf-8') as f:
                self.prompts = yaml.safe_load(f)
            logger.info("프롬프트 로드 완료")
        except Exception as e:
            logger.error(f"프롬프트 로드 실패: {e}")
            raise

    def _load_model(self, use_vllm=False):
        """Qwen 모델 로드"""
        logger.info(f"LLM 모델 로드 시작: {self.model_name}")

        try:
            if use_vllm:
                self.llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    dtype="float16",
                    gpu_memory_utilization=0.8,
                    max_model_len=512,
                    enforce_eager=False,
                    enable_prefix_caching=True,

                )

                self.sampling_params = SamplingParams(
                    temperature=0.2,
                    top_p=0.8,
                    max_tokens=256,
                    stop_token_ids=[self.llm.get_tokenizer().eos_token_id]
                )
            else:
                # Tokenizer 로드
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=False
                )

                # 모델 로드
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )

                # Pipeline 생성
                self.llm_pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer
                )

            logger.info("LLM 모델 로드 완료")

        except Exception as e:
            logger.error(f"LLM 모델 로드 실패: {e}")
            raise

    def get_prompt(self, key: str, **kwargs) -> str:
        """
        프롬프트 템플릿 가져오기

        Args:
            key: 프롬프트 키
            **kwargs: 템플릿에 주입할 변수

        Returns:
            완성된 프롬프트
        """
        template = self.prompts.get(key, {}).get('template', "")
        return template.format(**kwargs)

    def _extract_json(self, llm_output: str) -> Dict[str, Any]:
        """
        LLM 출력에서 JSON 추출

        Args:
            llm_output: LLM 원본 출력

        Returns:
            파싱된 JSON 딕셔너리
        """
        if not llm_output:
            return {}

        # JSON 블록 추출 (정규표현식)
        match = re.search(r'\{[\s\S]*\}', llm_output)
        if not match:
            logger.warning(f"JSON을 찾을 수 없음: {llm_output[:100]}")
            return {}

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return {}

    def classify_and_respond(
        self,
        user_text: str,
        current_step: str = ""
    ) -> Tuple[Intent, str]:
        """
        사용자 발화를 인텐트로 분류하고 응답 생성

        Args:
            user_text: 사용자 발화
            current_step: 현재 요리 단계 설명

        Returns:
            (Intent, 응답 텍스트)
        """
        logger.info(f"인텐트 분류 시작: '{user_text[:50]}...'")

        try:
            # 프롬프트 생성
            prompt = self.get_prompt(
                "unified_handler",
                text=user_text,
                current_step=current_step
            )

            # LLM 호출
            messages = [
                {"role": "system", "content": "너는 사용자의 요리 과정을 돕는 스마트 쉐프 조수야."},
                {"role": "user", "content": prompt}
            ]

            outputs = self.llm_pipe(
                messages,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.5,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # 결과 추출
            raw_output = outputs[0]['generated_text'][-1]['content'].strip()
            logger.debug(f"LLM 원본 출력: {raw_output}")

            # JSON 파싱
            data = self._extract_json(raw_output)

            # Intent 매핑
            raw_intent = data.get("Intent", "Out of Scope").strip()
            intent = self.INTENT_MAP.get(raw_intent, Intent.UNKNOWN)

            # Response 추출
            response = data.get("Response", "").strip()

            logger.info(f"인텐트 분류 완료: {intent} / '{response[:50]}...'")

            return intent, response

        except Exception as e:
            logger.error(f"인텐트 분류 실패: {e}")
            # 에러 시 UNKNOWN으로 반환
            return Intent.UNKNOWN, "죄송해요, 요청을 처리할 수 없어요."
