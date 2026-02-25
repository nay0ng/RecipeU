"""
텍스트 전처리 및 후처리 유틸리티

- TTS 전 텍스트 정제
- 특수문자 제거, 받침 처리 등
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# TTS 텍스트 정제
# ============================================================================

def clean_text_for_tts(text: str) -> str:
    """
    TTS 전 텍스트 정규화 및 정제

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    if not text:
        return ""

    # Unicode 정규화 (NFC)
    text = unicodedata.normalize('NFC', text)

    # 특수문자 제거 (마크다운, 강조 등)
    text = re.sub(r'[#*_\-]', '', text)

    # 받침 처리 (TTS 발음 개선)
    # 종성이 있는 한글을 발음이 자연스러운 형태로 변경
    replacement_map = {
        "읽": "익",
        "닭": "닥",
        "끓": "끌",
        "밝": "박",
        "젊": "점",
        "굵": "국",
        "삶": "삼",
        "옳": "올",
        "앓": "알",
        "넓": "널",
        "얇": "얄",
        "외곬": "외골",
        "핥": "할",
    }

    for old, new in replacement_map.items():
        text = text.replace(old, new)

    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ============================================================================
# 특수 문자 처리
# ============================================================================

def remove_markdown_syntax(text: str) -> str:
    """
    마크다운 문법 제거

    Args:
        text: 원본 텍스트

    Returns:
        마크다운 제거된 텍스트
    """
    # 코드 블록 제거
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)

    # 링크 제거 [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 강조 제거 **text** or *text* -> text
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)

    # 헤더 제거 ### text -> text
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    return text

# ============================================================================
# 숫자 처리
# ============================================================================

def normalize_numbers(text: str) -> str:
    """
    숫자를 한글로 변환 (선택적)

    Args:
        text: 원본 텍스트

    Returns:
        숫자가 한글로 변환된 텍스트
    """
    # 간단한 숫자 변환 (1-10)
    number_map = {
        "1": "일",
        "2": "이",
        "3": "삼",
        "4": "사",
        "5": "오",
        "6": "육",
        "7": "칠",
        "8": "팔",
        "9": "구",
        "10": "십"
    }

    # 단계 번호 변환 예: "3단계" -> "삼단계"
    for num, korean in number_map.items():
        text = re.sub(rf'\b{num}단계', f'{korean}단계', text)

    return text

# ============================================================================
# 전체 파이프라인
# ============================================================================

def preprocess_for_tts(text: str, remove_md: bool = True, normalize_num: bool = False) -> str:
    """
    TTS용 텍스트 전처리 파이프라인

    Args:
        text: 원본 텍스트
        remove_md: 마크다운 제거 여부
        normalize_num: 숫자 한글 변환 여부

    Returns:
        전처리된 텍스트
    """
    if not text:
        return ""

    # 1. 마크다운 제거
    if remove_md:
        text = remove_markdown_syntax(text)

    # 2. 숫자 정규화
    if normalize_num:
        text = normalize_numbers(text)

    # 3. TTS용 정제
    text = clean_text_for_tts(text)

    logger.debug(f"TTS 전처리 완료: '{text[:50]}...'")

    return text
