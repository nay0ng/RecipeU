"""
오디오 파일 처리 유틸리티

- 파일 검증
- 형식 변환
- 샘플링 레이트 변환
"""

import os
import logging
from typing import Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)

# ============================================================================
# 오디오 파일 검증
# ============================================================================

def validate_audio(file_path: str) -> bool:
    """
    오디오 파일 검증

    Args:
        file_path: 오디오 파일 경로

    Returns:
        유효 여부
    """
    if not os.path.exists(file_path):
        logger.error(f"파일이 존재하지 않음: {file_path}")
        return False

    # 확장자 확인
    valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in valid_extensions:
        logger.error(f"지원하지 않는 확장자: {ext}")
        return False

    # 파일 크기 확인 (0 바이트 아닌지)
    if os.path.getsize(file_path) == 0:
        logger.error(f"빈 파일: {file_path}")
        return False

    return True

# ============================================================================
# 오디오 변환
# ============================================================================

def convert_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000
) -> str:
    """
    오디오 파일을 WAV로 변환

    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로 (None이면 자동 생성)
        sample_rate: 목표 샘플링 레이트

    Returns:
        변환된 WAV 파일 경로
    """
    if not validate_audio(input_path):
        raise ValueError(f"유효하지 않은 오디오 파일: {input_path}")

    # 출력 경로 생성
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_converted.wav"

    try:
        # 오디오 로드
        waveform, orig_sr = torchaudio.load(input_path)

        # 스테레오 -> 모노
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 리샘플링
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)

        # WAV 저장
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torchaudio.save(output_path, waveform, sample_rate)

        logger.info(f"WAV 변환 완료: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"WAV 변환 실패: {e}")
        raise

# ============================================================================
# 리샘플링
# ============================================================================

def resample_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 16000
) -> str:
    """
    오디오 파일 리샘플링

    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로
        target_sr: 목표 샘플링 레이트

    Returns:
        리샘플링된 파일 경로
    """
    try:
        waveform, orig_sr = torchaudio.load(input_path)

        if orig_sr == target_sr:
            logger.info(f"이미 {target_sr}Hz: {input_path}")
            return input_path

        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torchaudio.save(output_path, waveform, target_sr)

        logger.info(f"리샘플링 완료: {orig_sr}Hz -> {target_sr}Hz")
        return output_path

    except Exception as e:
        logger.error(f"리샘플링 실패: {e}")
        raise

# ============================================================================
# 스테레오 -> 모노
# ============================================================================

def stereo_to_mono(
    input_path: str,
    output_path: str
) -> str:
    """
    스테레오를 모노로 변환

    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로

    Returns:
        변환된 파일 경로
    """
    try:
        waveform, sr = torchaudio.load(input_path)

        if waveform.shape[0] == 1:
            logger.info(f"이미 모노: {input_path}")
            return input_path

        # 평균으로 모노 변환
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torchaudio.save(output_path, waveform, sr)

        logger.info(f"모노 변환 완료: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"모노 변환 실패: {e}")
        raise

# ============================================================================
# 오디오 정보 조회
# ============================================================================

def get_audio_info(file_path: str) -> dict:
    """
    오디오 파일 정보 조회

    Args:
        file_path: 오디오 파일 경로

    Returns:
        오디오 정보 딕셔너리
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일이 존재하지 않음: {file_path}")

    try:
        waveform, sr = torchaudio.load(file_path)

        info = {
            "path": file_path,
            "sample_rate": sr,
            "channels": waveform.shape[0],
            "duration_sec": waveform.shape[1] / sr,
            "num_samples": waveform.shape[1],
        }

        return info

    except Exception as e:
        logger.error(f"오디오 정보 조회 실패: {e}")
        raise
