"""
services/rag_neo4j.py
rag.py에서 Milvus → Neo4j로 교체한 버전

변경:
  - Milvus, ClovaXEmbeddings 제거
  - Neo4j 검색으로 대체
  - 알레르기/조리도구 필터 추가 (기존에 없던 기능)

유지:
  - ClovaStudioReranker (그대로)
  - generate_answer, generate_recipe_json (그대로)
  - ChatClovaX, query() (그대로)
"""

import json
import os
import http.client
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from neo4j import GraphDatabase

try:
    from langchain_naver import ChatClovaX
except ImportError:
    from langchain_community.chat_models import ChatClovaX

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import time

load_dotenv()


def _t():
    return time.time()

def _log_step(label: str, start: float, end: float):
    print(f"  ⏱️  [{label}] {end - start:.1f}초")


# ─────────────────────────────────────────────
# ClovaStudioReranker (기존 그대로)
# ─────────────────────────────────────────────
class ClovaStudioReranker:
    def __init__(self, api_key: str, request_id: str = "recipe-rag-rerank"):
        self.host = 'clovastudio.stream.ntruss.com'
        self.api_key = f'Bearer {api_key}'
        self.request_id = request_id

    def rerank(self, query: str, documents: List[Dict[str, str]], max_tokens: int = 1024) -> Dict:
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self.api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id
        }
        request_data = {"documents": documents, "query": query, "maxTokens": max_tokens}
        try:
            conn = http.client.HTTPSConnection(self.host)
            conn.request('POST', '/v1/api-tools/reranker', json.dumps(request_data), headers)
            response = conn.getresponse()
            result = json.loads(response.read().decode(encoding='utf-8'))
            conn.close()
            if result.get('status', {}).get('code') == '20000':
                return result.get('result', {})
            else:
                print(f"[WARNING] Reranker API 오류: {result}")
                return None
        except Exception as e:
            print(f"[ERROR] Reranker API 호출 실패: {e}")
            return None


# ─────────────────────────────────────────────
# RecipeRAGLangChain (Neo4j 버전)
# ─────────────────────────────────────────────
class RecipeRAGLangChain:

    def __init__(
        self,
        use_reranker: bool = True,
        chat_model: str = "HCX-DASH-001",
        temperature: float = 0,
        max_tokens: int = 2000,
    ):
        print("\n" + "="*60)
        print("Recipe RAG System (LangChain + CLOVA X + Neo4j)")
        print("="*60)

        # 1. ChatClovaX 초기화
        print(f"\n[1/3] CLOVA X Chat 모델 초기화 중 (model: {chat_model})")
        self.chat_model = ChatClovaX(
            model=chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print("[OK] Chat 모델 초기화 완료")

        # 2. Reranker 초기화
        print("\n[2/3] CLOVA Studio Reranker 초기화 중")
        self.reranker = None
        self.use_reranker = use_reranker
        if use_reranker:
            api_key = os.getenv("CLOVASTUDIO_RERANKER_API_KEY")
            request_id = os.getenv("CLOVASTUDIO_REQUEST_ID", "recipe-rag-rerank")
            if api_key:
                self.reranker = ClovaStudioReranker(api_key=api_key, request_id=request_id)
                print("[OK] CLOVA Studio Reranker 활성화")
            else:
                print("[WARNING] CLOVASTUDIO_RERANKER_API_KEY 없음. Reranker 비활성화.")
                self.use_reranker = False

        # 3. Neo4j 연결
        print("\n[3/3] Neo4j 연결 중")
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        with self.neo4j_driver.session() as session:
            count = session.run("MATCH (r:Recipe) RETURN count(r) AS cnt").single()["cnt"]
            print(f"[OK] Neo4j 연결 성공 (레시피 {count:,}개)")

        print("\n" + "="*60)
        print("시스템 초기화 완료")
        print("="*60 + "\n")

    # ─────────────────────────────────────────────
    # Neo4j 검색 (Milvus 검색 대체)
    # allergies: 제외할 재료 리스트 ["새우", "땅콩"]
    # user_tools: 사용자가 가진 특별 도구 ["에어프라이어", "오븐"]
    # ─────────────────────────────────────────────
    def _neo4j_search(
        self,
        query: str,
        k: int,
        allergies: List[str] = None,
        user_tools: List[str] = None,
    ) -> List[tuple]:
        t_start = _t()
        results = []
        seen_ids = set()

        # 알레르기 필터 Cypher 조건
        allergy_clause = ""
        if allergies:
            conditions = " AND ".join([f'NOT (r)-[:CONTAINS]->(:Ingredient {{name: "{a}"}})' for a in allergies])
            allergy_clause = f"AND {conditions}"

        # 조리도구 필터 Cypher 조건
        # user_tools가 있으면: 레시피에 필요한 특별 도구가 모두 user_tools 안에 있어야 함
        # user_tools가 없으면: 필터 없음 (전체)
        tool_clause = ""
        if user_tools:
            tool_clause = f"AND ALL(tool IN r.cooking_tools WHERE tool IN {json.dumps(user_tools, ensure_ascii=False)})"

        with self.neo4j_driver.session() as session:

            # 1단계: 제목 키워드 매칭
            title_query = f"""
                MATCH (r:Recipe)
                WHERE r.title CONTAINS $query
                {allergy_clause}
                {tool_clause}
                RETURN r.id AS recipe_id, r.title AS title, r.intro AS intro,
                       r.cook_time AS cook_time, r.level AS level,
                       r.author AS author, r.detail_url AS source,
                       r.image AS image, r.steps AS steps,
                       r.cooking_tools AS cooking_tools
                LIMIT $k
            """
            title_results = session.run(title_query, {"query": query, "k": k})
            for record in title_results:
                rid = record["recipe_id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    results.append((self._record_to_document(record), 0.0))

            print(f"  🎯 [제목 매칭] {len(results)}개")

            # 2단계: 부족하면 재료명으로 보완
            if len(results) < k:
                remaining = k - len(results)
                ingredient_query = f"""
                    MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient)
                    WHERE i.name CONTAINS $query
                    {allergy_clause}
                    {tool_clause}
                    WITH r, count(i) AS match_count
                    ORDER BY match_count DESC
                    RETURN r.id AS recipe_id, r.title AS title, r.intro AS intro,
                           r.cook_time AS cook_time, r.level AS level,
                           r.author AS author, r.detail_url AS source,
                           r.image AS image, r.steps AS steps,
                           r.cooking_tools AS cooking_tools
                    LIMIT $remaining
                """
                ing_results = session.run(ingredient_query, {"query": query, "remaining": remaining})
                added = 0
                for record in ing_results:
                    rid = record["recipe_id"]
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        results.append((self._record_to_document(record), 1.0))
                        added += 1
                if added > 0:
                    print(f"  🥕 [재료 매칭] {added}개 추가")

        _log_step("Neo4j 검색", t_start, _t())
        return results

    def _record_to_document(self, record) -> Document:
        title = record["title"] or "N/A"
        intro = record["intro"] or ""
        steps = record["steps"] or ""
        # page_content에 steps까지 포함 → LLM context로 바로 활용
        page_content = f"{title}\n{intro}\n\n{steps}"

        return Document(
            page_content=page_content,
            metadata={
                "title":         title,
                "author":        record["author"] or "N/A",
                "source":        record["source"] or "N/A",
                "cook_time":     record["cook_time"] or "N/A",
                "level":         record["level"] or "N/A",
                "recipe_id":     record["recipe_id"] or "",
                "image_url":     record["image"] or "",
                "cooking_tools": record["cooking_tools"] or [],
            }
        )

    def _rerank_documents(self, query: str, documents: List[Document], top_n: int = 5) -> List[tuple]:
        if not self.reranker or not documents:
            return [(doc, 1.0) for doc in documents[:top_n]]

        rerank_docs = [{"id": f"doc{i}", "doc": doc.page_content[:2000]} for i, doc in enumerate(documents)]
        result = self.reranker.rerank(query, rerank_docs, max_tokens=1024)

        if not result:
            print("[WARNING] Reranker 실패, 원본 순서 사용")
            return [(doc, 1.0) for doc in documents[:top_n]]

        reranked = []
        for item in result.get('topPassages', [])[:top_n]:
            try:
                idx = int(item.get('id', '').replace('doc', ''))
                if 0 <= idx < len(documents):
                    reranked.append((documents[idx], float(item.get('score', 0.0))))
            except (ValueError, IndexError):
                continue

        return reranked if reranked else [(doc, 1.0) for doc in documents[:top_n]]

    # ─────────────────────────────────────────────
    # search_recipes - allergies, user_tools 파라미터 추가
    # ─────────────────────────────────────────────
    def search_recipes(
        self,
        query: str,
        k: int = 3,
        use_rerank: bool = False,
        allergies: List[str] = None,       # ← 신규: 알레르기 재료
        user_tools: List[str] = None,      # ← 신규: 사용자 보유 도구
    ) -> List[Dict]:

        t_total_start = _t()
        use_rerank = use_rerank if use_rerank is not None else self.use_reranker
        print(f"\n  📍 [search_recipes] 시작 (k={k}, rerank={use_rerank})")
        if allergies:
            print(f"  🚫 알레르기 필터: {allergies}")
        if user_tools:
            print(f"  🔧 도구 필터: {user_tools}")

        if use_rerank and self.reranker:
            search_k = min(k * 3, 20)
            docs_with_scores = self._neo4j_search(query, search_k, allergies, user_tools)
            docs = [doc for doc, _ in docs_with_scores]
            vector_scores = {id(doc): float(score) for doc, score in docs_with_scores}

            t_rerank_start = _t()
            reranked_results = self._rerank_documents(query, docs, top_n=k)
            _log_step("Reranker API", t_rerank_start, _t())

            results = []
            for doc, rerank_score in reranked_results:
                results.append({
                    "content":       doc.page_content,
                    "vector_score":  vector_scores.get(id(doc), 0.0),
                    "rerank_score":  float(rerank_score),
                    "title":         doc.metadata.get("title", "N/A"),
                    "author":        doc.metadata.get("author", "N/A"),
                    "source":        doc.metadata.get("source", "N/A"),
                    "cook_time":     doc.metadata.get("cook_time", "N/A"),
                    "level":         doc.metadata.get("level", "N/A"),
                    "recipe_id":     doc.metadata.get("recipe_id", "N/A"),
                    "image":         doc.metadata.get("image_url", ""),
                    "cooking_tools": doc.metadata.get("cooking_tools", []),
                })
        else:
            docs_with_scores = self._neo4j_search(query, k, allergies, user_tools)
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content":       doc.page_content,
                    "vector_score":  float(score),
                    "title":         doc.metadata.get("title", "N/A"),
                    "author":        doc.metadata.get("author", "N/A"),
                    "source":        doc.metadata.get("source", "N/A"),
                    "cook_time":     doc.metadata.get("cook_time", "N/A"),
                    "level":         doc.metadata.get("level", "N/A"),
                    "recipe_id":     doc.metadata.get("recipe_id", "N/A"),
                    "image":         doc.metadata.get("image_url", ""),
                    "cooking_tools": doc.metadata.get("cooking_tools", []),
                })

        _log_step("search_recipes 합계", t_total_start, _t())
        print(f"  📍 [search_recipes] 완료\n")
        return results

    # ─────────────────────────────────────────────
    # generate_answer (기존 그대로)
    # ─────────────────────────────────────────────
    def generate_answer(self, query: str, context_docs: List[Dict], system_prompt: Optional[str] = None) -> str:
        print(f"  📍 [generate_answer] 시작")
        t_total_start = _t()

        if system_prompt is None:
            system_prompt = """당신은 한국 요리 전문가이자 친절한 레시피 어시스턴트입니다.

# 🚨 절대 규칙
1. **반드시 하나의 요리만 추천하세요!**
2. **여러 요리를 리스트로 나열하지 마세요!**
3. **조리법은 1~2줄로 간단히!**

# 필수 답변 형식
오늘의 추천 요리는 [요리명] 입니다.

**재료 (N인분, 조리시간):**
- 주요 재료 5~7개만 간단히 나열

**조리법:**
1~2줄로 핵심만 요약

**특징:**
한 줄로 이 요리의 매력 설명

{context}"""

        documents = [
            Document(
                page_content=d.get("content", ""),
                metadata={"title": d.get("title", "N/A"), "author": d.get("author", "N/A"), "source": d.get("source", "N/A")}
            ) for d in context_docs
        ]
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        chain = create_stuff_documents_chain(self.chat_model, prompt)

        t_llm_start = _t()
        try:
            result = chain.invoke({"input": query, "context": documents})
            _log_step("LLM 호출 (generate_answer)", t_llm_start, _t())
            _log_step("generate_answer 합계", t_total_start, _t())
            return result
        except Exception as e:
            print(f"답변 생성 오류: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    # ─────────────────────────────────────────────
    # generate_recipe_json (기존 그대로)
    # ─────────────────────────────────────────────
    def generate_recipe_json(
        self,
        user_message: str,
        context_docs: List[Dict],
        constraints_text: str = "",
        conversation_history: str = "",
        system_prompt: Optional[str] = None,
    ) -> dict:
        print(f"  📍 [generate_recipe_json] 시작")
        t_total_start = _t()

        if system_prompt is None:
            system_prompt = """당신은 한국 요리 전문가입니다.

# 역할
주어진 레시피 데이터베이스와 **대화 히스토리**를 참고하여 사용자가 원하는 요리 레시피를 JSON 형식으로 **상세하게** 생성해주세요.

# 사용자 제약사항
{constraints_text}

# 대화 히스토리 (매우 중요!)
{conversation_history}

**중요**:
- 대화 히스토리를 꼼꼼히 읽고 사용자의 모든 요구사항을 반영하세요
- 알레르기와 비선호 재료는 절대 사용하지 마세요

# 출력 형식
반드시 다음 JSON 형식만 출력하고, 다른 설명은 붙이지 마세요:
{{{{
"title": "요리 이름",
"intro": "한 줄 소개",
"cook_time": "예: 10~15분",
"level": "예: 초급",
"servings": "예: 2인분",
"ingredients": [
    {{{{"name": "재료명", "amount": "양", "note": "선택사항"}}}}
],
"steps": [
    {{{{"no": 1, "desc": "구체적이고 상세한 설명"}}}},
    {{{{"no": 2, "desc": "..."}}}}
],
}}}}

{{context}}"""

        formatted_system_prompt = system_prompt.format(
            constraints_text=constraints_text if constraints_text else "없음",
            conversation_history=conversation_history if conversation_history else "없음",
            context="{context}"
        )
        documents = [
            Document(
                page_content=d.get("content", ""),
                metadata={"title": d.get("title", "N/A"), "author": d.get("author", "N/A"), "source": d.get("source", "N/A")}
            ) for d in context_docs
        ]
        prompt = ChatPromptTemplate.from_messages([("system", formatted_system_prompt), ("human", "{input}")])
        chain = create_stuff_documents_chain(self.chat_model, prompt)

        t_llm_start = _t()
        try:
            result = chain.invoke({"input": user_message, "context": documents})
            _log_step("LLM 호출 (generate_recipe_json)", t_llm_start, _t())
            response_text = result if isinstance(result, str) else str(result)
        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return self._get_default_recipe()

        try:
            clean = response_text.strip()
            if clean.startswith("```json"): clean = clean[7:]
            if clean.startswith("```"):     clean = clean[3:]
            if clean.endswith("```"):       clean = clean[:-3]
            parsed = json.loads(clean.strip())
            print(f"✅ 레시피 JSON 생성 성공: {parsed.get('title', 'N/A')}")
            _log_step("generate_recipe_json 합계", t_total_start, _t())
            return parsed
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return self._get_default_recipe()

    def _get_default_recipe(self) -> dict:
        return {
            "title": "레시피 생성 실패",
            "intro": "레시피를 생성하는 중 오류가 발생했습니다.",
            "cook_time": "N/A", "level": "N/A", "servings": "N/A",
            "ingredients": [], "steps": [],
        }

    # ─────────────────────────────────────────────
    # query (기존 그대로 + allergies/user_tools 파라미터 추가)
    # ─────────────────────────────────────────────
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_rerank: bool = None,
        return_references: bool = True,
        allergies: List[str] = None,       # ← 신규
        user_tools: List[str] = None,      # ← 신규
    ) -> Dict[str, Any]:
        print(f"\n{'='*50}")
        print(f"  🔍 [query] 시작: \"{question[:40]}...\"")
        print(f"{'='*50}")
        t_query_start = _t()

        retrieved_docs = self.search_recipes(
            question, k=top_k, use_rerank=use_rerank,
            allergies=allergies, user_tools=user_tools
        )
        answer = self.generate_answer(question, retrieved_docs)

        result = {"question": question, "answer": answer}
        if return_references:
            result["references"] = retrieved_docs
            result["num_references"] = len(retrieved_docs)

        _log_step("query() 전체 합계", t_query_start, _t())
        print(f"{'='*50}\n")
        return result

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()