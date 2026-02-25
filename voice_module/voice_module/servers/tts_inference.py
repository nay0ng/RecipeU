"""
GPT-SoVITS TTS 추론 모듈
- 공식 GPT-SoVITS TTS_infer_pack/TTS.py 기반
- 다양한 톤의 레퍼런스 오디오 지원
- 문장 입력 → 음성 파일 반환

사용법:
    from tts_inference import TTSInference

    tts = TTSInference()
    tts.register_default_references()
    audio_path = tts.synthesize("안녕하세요", tone="bright") <- 함수 변경해서 확인하세요! 자세한 건 주석 참고

폴더 구조:
    WORK_DIR/
    ├── tts_inference.py
    ├── references/
    ├── outputs/
    └── GPT-SoVITS/
        ├── GPT_SoVITS/
        │   ├── TTS_infer_pack/
        │   ├── AR/
        │   ├── module/
        │   └── pretrained_models/
        ├── GPT_weights_v2/
        └── SoVITS_weights_v2/
"""

import os
import sys
import numpy as np
import soundfile as sf
from typing import Optional, Literal, Tuple
from datetime import datetime

# ============================================================================
# 경로 설정 (상대 경로 기반)
# ============================================================================

# 현재 파일(tts_inference.py)의 디렉토리 기준으로 상대 경로 계산
# servers/tts_inference.py → 프로젝트 루트는 한 단계 위
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.dirname(_CURRENT_DIR)  # servers/ 의 상위 = 프로젝트 루트

# utils 폴더 (GPT-SoVITS, references가 있는 곳)
UTILS_DIR = os.path.join(WORK_DIR, "utils")

# GPT-SoVITS 루트 (utils 폴더 내)
GPT_SOVITS_ROOT = os.path.join(UTILS_DIR, "GPT-SoVITS")

# GPT_SoVITS 모듈 경로 (AR, BigVGAN, TTS_infer_pack 등이 있는 곳)
GPT_SOVITS_MODULE = os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS")

# 학습된 모델 경로
TRAINED_MODELS_DIR = GPT_SOVITS_ROOT

# 레퍼런스 오디오 경로 (utils 폴더 내)
REFERENCES_DIR = os.path.join(UTILS_DIR, "references")

# 출력 폴더 경로
OUTPUTS_DIR = os.path.join(WORK_DIR, "outputs")

# 모듈 경로 추가
sys.path.insert(0, GPT_SOVITS_ROOT)
sys.path.insert(0, GPT_SOVITS_MODULE)

# sv.py가 필요로 하는 eres2net 경로 추가
sys.path.insert(0, os.path.join(GPT_SOVITS_MODULE, "eres2net"))

# 작업 디렉토리 변경 (상대 경로 참조를 위해)
os.chdir(WORK_DIR)

# TTS_infer_pack은 GPT_SoVITS 안에 있음
from TTS_infer_pack.TTS import TTS, TTS_Config


# 톤 타입 정의
ToneType = Literal["bright", "calm", "serious", "excited"]


# ============================================================================
# 기본 레퍼런스 설정 (13개)
# ============================================================================

DEFAULT_REFERENCES = {
    "bright": ("1.bright.wav", "오늘은 아주 맛있는 감자수프를 만들어볼게요."),
    "calm": ("2.calm.wav", "오늘은 아주 맛있는 감자수프를 함께 만들어보도록 하겠습니다."),
    "excited": ("3.excited.wav", "오늘은 아주 맛있는 감자수프를 한 번 만들어봐요."),
    "serious": ("4.serious.wav", "오늘은 아주 맛있는 감자수프를 한 번 만들어보도록 할게요."),
    "conan": ("5.conan.wav", "안녕하세요. 제 이름은 코난, 탐정이죠."),
    "jjanggu": ("6.jjanggu.wav", "안녕, 나는 짱구!"),
    "keroro": ("7.keroro.wav", "저는 케롱별에서 온 케로로 중사라고 합니다."),
    "dunyarzad": ("8.Dunyarzad.wav", "응, 고마워, 페이몬. 화신 탄신 축제에 참가한 사람들 모두 재밌게 즐겼으면 좋겠어"),
    "kazuha": ("9.Kaedehara_Kazuha.wav", "최선을 다할테니 걱정하지 마세요."),
    "kenji": ("10.Kenji.wav", "오늘의 레시피는 감자수프야."),
    "kiwi": ("11.Kiwi.wav", "오늘의 레시피는 감자수프야."),
    "luo_qiao": ("12.Luo_Qiao.wav", "투자라면… 최근 크게 화제 되고 있는 반딧불 정수 사업 말씀인가요?"),
    "paimon": ("13.Paimon.wav", "와, 이게 휴식을 즐기는 캐서린인가? 정말 달라 보여."),
}


class ReferenceAudio:
    """레퍼런스 오디오 정보를 담는 클래스"""
    def __init__(self, path: str, text: str, lang: str = "ko"):
        self.path = path
        self.text = text
        self.lang = lang


class TTSInference:
    """
    GPT-SoVITS 기반 TTS 추론 클래스

    Args:
        gpt_model_path: GPT 모델 경로 (.ckpt)
        sovits_model_path: SoVITS 모델 경로 (.pth)
        device: 사용할 디바이스 ("cuda" 또는 "cpu")
        is_half: half precision 사용 여부 (GPU에서만 가능)
    """

    # 톤별 레퍼런스 오디오 (인스턴스 생성 시 초기화)
    TONE_REFERENCES = {}

    def __init__(
        self,
        gpt_model_path: str = None,
        sovits_model_path: str = None,
        device: str = "cuda",
        is_half: bool = True,
    ):
        self.device = device
        self.is_half = is_half

        # 기본 모델 경로 설정 (학습된 모델 - 작업 폴더 내)
        if gpt_model_path is None:
            gpt_model_path = os.path.join(TRAINED_MODELS_DIR, "GPT_weights_v2/hj-voice-e15.ckpt")
        if sovits_model_path is None:
            sovits_model_path = os.path.join(TRAINED_MODELS_DIR, "SoVITS_weights_v2/hj-voice_e8_s72.pth")

        # pretrained_models 경로 (GPT_SoVITS 모듈 내부)
        cnhuhbert_path = os.path.join(GPT_SOVITS_MODULE, "pretrained_models/chinese-hubert-base")
        bert_path = os.path.join(GPT_SOVITS_MODULE, "pretrained_models/chinese-roberta-wwm-ext-large")

        # TTS 설정 생성
        config_dict = {
            "device": device,
            "is_half": is_half,
            "version": "v2",
            "t2s_weights_path": gpt_model_path,
            "vits_weights_path": sovits_model_path,
            "cnhuhbert_base_path": cnhuhbert_path,
            "bert_base_path": bert_path,
        }

        # TTS 파이프라인 초기화
        tts_config = TTS_Config({"custom": config_dict})
        self.tts_pipeline = TTS(tts_config)

        # 인스턴스별 레퍼런스 딕셔너리 초기화
        self.TONE_REFERENCES = {}

    def register_default_references(self):
        """DEFAULT_REFERENCES에 정의된 기본 레퍼런스들을 등록"""
        registered = 0
        for name, (filename, text) in DEFAULT_REFERENCES.items():
            path = os.path.join(REFERENCES_DIR, filename)
            if os.path.exists(path):
                self.set_reference(tone=name, audio_path=path, text=text, lang="ko")
                registered += 1
        return registered

    def set_reference(self, tone: ToneType, audio_path: str, text: str, lang: str = "ko"):
        """
        특정 톤의 레퍼런스 오디오 설정

        Args:
            tone: 톤 타입 ("bright", "calm", "serious", "excited")
            audio_path: 레퍼런스 오디오 파일 경로
            text: 레퍼런스 오디오의 텍스트 내용
            lang: 언어 코드 (기본: "ko")
        """
        self.TONE_REFERENCES[tone] = ReferenceAudio(
            path=audio_path,
            text=text,
            lang=lang
        )

    def get_reference(self, tone: ToneType) -> ReferenceAudio:
        """톤에 해당하는 레퍼런스 오디오 정보 반환"""
        if tone not in self.TONE_REFERENCES:
            raise ValueError(f"지원하지 않는 톤입니다: {tone}. 가능한 톤: {list(self.TONE_REFERENCES.keys())}")
        return self.TONE_REFERENCES[tone]

    def synthesize(
        self,
        text: str,
        tone: ToneType = "calm",
        output_path: Optional[str] = None,
        text_lang: str = "ko",
        speed_factor: float = 1.0,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        batch_size: int = 1,
        seed: int = -1,
    ) -> str:
        """
        [Blocking] 텍스트를 음성으로 변환하여 '파일(.wav)'로 저장합니다.

        전체 음성이 완성될 때까지 기다린 후(Blocking), 완성된 파일을 디스크에 쓰고
        그 경로를 반환합니다.

        용도:
            - 생성된 음성을 영구적으로 보관해야 할 때
            - 로그 남기기 또는 디버깅용
            - 빠른 응답 속도보다 데이터의 안정성이 중요할 때

        Args:
            text (str): 합성할 텍스트 내용
            tone (ToneType): 사용할 목소리 톤 ('bright', 'calm' 등)
            output_path (str, optional): 저장할 파일 경로. None이면 outputs 폴더에 자동 생성.
            speed_factor (float): 말하기 속도 (기본 1.0)
            top_k (int): 다음 토큰 예측 후보 수 (낮을수록 빠르고 안정적)

        Returns:
            str: 저장된 오디오 파일의 절대 경로
        """
        # 레퍼런스 오디오 가져오기
        ref = self.get_reference(tone)

        # 레퍼런스 파일 존재 확인
        if not os.path.exists(ref.path):
            raise FileNotFoundError(f"레퍼런스 오디오 파일을 찾을 수 없습니다: {ref.path}")

        # 출력 경로 자동 생성
        if output_path is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUTS_DIR, f"tts_{tone}_{timestamp}.wav")

        # 추론 요청 파라미터 구성
        req = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref.path,
            "prompt_text": ref.text,
            "prompt_lang": ref.lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": "cut5",
            "batch_size": batch_size,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": speed_factor,
            "fragment_interval": 0.3,
            "seed": seed,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "streaming_mode": False,
            "return_fragment": False,
        }

        # 추론 실행
        audio_chunks = []
        sample_rate = None

        for sr, audio_chunk in self.tts_pipeline.run(req):
            sample_rate = sr
            audio_chunks.append(audio_chunk)

        # 오디오 청크 합치기
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
        else:
            raise RuntimeError("음성 합성에 실패했습니다.")

        # 파일 저장
        sf.write(output_path, audio_data, sample_rate)

        return output_path


    def synthesize_to_bytes(
        self,
        text: str,
        tone: ToneType = "calm",
        text_lang: str = "ko",
        speed_factor: float = 1.0,
        **kwargs
    ) -> Tuple[int, np.ndarray]:
        """
        [Non-File] 텍스트를 음성으로 변환하여 '메모리 상의 데이터(NumPy)'로 반환합니다.

        파일 시스템(HDD/SSD)을 거치지 않고 RAM에서만 처리하므로 `synthesize`보다 빠릅니다.
        완성된 전체 오디오 데이터를 한 번에 반환합니다.

        용도:
            - 일반적인 API 응답 (파일 저장 없이 바이너리 전송)
            - 생성된 오디오를 즉시 다른 모델의 입력으로 쓸 때
            - 짧은 문장을 빠르게 처리할 때

        Args:
            text (str): 합성할 텍스트
            tone (ToneType): 목소리 톤
            **kwargs: top_k, top_p 등의 추가 추론 파라미터

        Returns:
            Tuple[int, np.ndarray]: (샘플 레이트, 오디오 데이터 배열)
        """
        ref = self.get_reference(tone)

        if not os.path.exists(ref.path):
            raise FileNotFoundError(f"레퍼런스 오디오 파일을 찾을 수 없습니다: {ref.path}")

        req = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref.path,
            "prompt_text": ref.text,
            "prompt_lang": ref.lang,
            "top_k": kwargs.get("top_k", 15),
            "top_p": kwargs.get("top_p", 1.0),
            "temperature": kwargs.get("temperature", 1.0),
            "text_split_method": "cut5",
            "batch_size": kwargs.get("batch_size", 1),
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": speed_factor,
            "fragment_interval": 0.3,
            "seed": kwargs.get("seed", -1),
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "streaming_mode": False,
            "return_fragment": False,
        }

        audio_chunks = []
        sample_rate = None

        for sr, audio_chunk in self.tts_pipeline.run(req):
            sample_rate = sr
            audio_chunks.append(audio_chunk)

        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
            return sample_rate, audio_data
        else:
            raise RuntimeError("음성 합성에 실패했습니다.")

    
    
    # !!!!!!!!!stream 처리로 추가됨
    def synthesize_stream_generator(
            self,
            text: str,
            tone: ToneType = "calm",
            text_lang: str = "ko",
            speed_factor: float = 1.2,
            **kwargs
        ):
            """
            [Streaming] 음성 데이터를 조각(Chunk) 단위로 실시간 반환하는 제너레이터입니다.

            전체 문장이 완성되기를 기다리지 않고, 앞부분이 생성되는 즉시 `yield`로 반환합니다.
            클라이언트가 첫 소리를 듣기까지의 시간(TTFB)을 획기적으로 줄여줍니다.

            용도:
                - 실시간 대화형 AI 서비스 (Live Chat)
                - 긴 문장을 읽어줄 때 (사용자 대기 시간 최소화)
                - FastAPI의 StreamingResponse와 연동

            Args:
                text (str): 합성할 텍스트
                tone (ToneType): 목소리 톤
                speed_factor (float): 말하기 속도
                **kwargs: 내부적으로 streaming_mode=True, return_fragment=True가 강제 적용됨

            Yields:
                Tuple[int, np.ndarray]: (샘플 레이트, 짧은 오디오 조각 배열)
            """
            ref = self.get_reference(tone)

            if not os.path.exists(ref.path):
                raise FileNotFoundError(f"레퍼런스 오디오 파일을 찾을 수 없습니다: {ref.path}")

            # 스트리밍을 위한 요청 설정
            req = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": ref.path,
                "prompt_text": ref.text,
                "prompt_lang": ref.lang,
                "top_k": kwargs.get("top_k", 5),        # 속도를 위해 5로 낮춤 (기존 15)
                "top_p": kwargs.get("top_p", 1.0),
                "temperature": kwargs.get("temperature", 1.0),
                "text_split_method": "cut5",            # 문장 분리 방식
                "batch_size": 1,
                "speed_factor": speed_factor,
                "fragment_interval": 0.3,
                "seed": -1,
                "parallel_infer": True,
                "repetition_penalty": 1.35,
                "streaming_mode": True,                 # [중요] 스트리밍 모드 켜기
                "return_fragment": True,                # [중요] 조각 단위 반환
            }
            
            # Generator 실행 (yield)
            for sr, audio_chunk in self.tts_pipeline.run(req):
                # audio_chunk는 numpy array입니다.
                # 이를 바로 yield 하거나 bytes로 변환해서 넘길 수 있습니다.
                yield sr, audio_chunk

