# core/dependencies.py
"""
FastAPI 의존성 관리
"""
import logging
from typing import Optional

from services.rag_neo4j_option_b import RecipeRAGLangChain  # [옵션 B] 1단계: Direct Cypher
from app.config import settings

logger = logging.getLogger(__name__)

# 싱글톤 인스턴스
_rag_system: Optional[RecipeRAGLangChain] = None
_rag_init_attempted: bool = False


def get_rag_system() -> Optional[RecipeRAGLangChain]:
    """RAG 시스템 싱글톤 - 초기화 실패 시 None 반환 (캐싱하지 않음)"""
    global _rag_system, _rag_init_attempted

    if _rag_init_attempted:
        return _rag_system

    _rag_init_attempted = True

    if not settings.CLOVASTUDIO_API_KEY:
        logger.error(
            "RAG 시스템 초기화 실패: CLOVASTUDIO_API_KEY 환경변수가 설정되지 않았습니다."
        )
        return None

    if not settings.NEO4J_URI:
        logger.error(
            "RAG 시스템 초기화 실패: NEO4J_URI 환경변수가 설정되지 않았습니다."
        )
        return None

    try:
        _rag_system = RecipeRAGLangChain(
            use_reranker=settings.USE_RERANKER,
            temperature=0.2,
            max_tokens=2000,
        )
        logger.info("RAG 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"RAG 시스템 초기화 실패: {e}", exc_info=True)
        _rag_system = None

    return _rag_system
