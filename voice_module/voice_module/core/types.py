"""
공통 타입 정의
"""

from enum import Enum

class Intent(str, Enum):
    """요리 세션 인텐트"""
    NEXT = "next_step"
    PREV = "prev_step"
    FINISH = "finish"
    SUB_ING = "substitute_ingredient"
    SUB_TOOL = "substitute_tool"
    FAILURE = "failure"
    UNKNOWN = "unknown"

# Intent 매핑 (LLM 출력 → Intent Enum)
INTENT_MAP = {
    "Next": Intent.NEXT,
    "Prev": Intent.PREV,
    "Finish": Intent.FINISH,
    "Missing Ingredient": Intent.SUB_ING,
    "Missing Tool": Intent.SUB_TOOL,
    "Failure": Intent.FAILURE,
    "Out of Scope": Intent.UNKNOWN,
}
