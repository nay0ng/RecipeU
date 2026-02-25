def print_token_summary():
    """누적된 토큰 사용량 요약 출력 (마크다운 형식)"""
    if _token_accumulator["total"] == 0:
        return

    print(f"\n{'🔷'*30}")
    print(f"{'  '*10}📊 전체 토큰 사용량 요약")
    print(f"{'🔷'*30}")
    print(f"📥 총 입력 토큰 (prompt):     {_token_accumulator['prompt']:,} tokens")
    print(f"📤 총 출력 토큰 (completion): {_token_accumulator['completion']:,} tokens")
    print(f"📊 총합 (total):              {_token_accumulator['total']:,} tokens")
    print(f"{'🔷'*30}\n")

    # ✅ 1) 노드별 토큰/시간 요약 표 (마크다운)
    print("\n" + "="*100)
    print("📋 노드별 상세 요약")
    print("="*100)
    print("| Step | Node | 설명 | Prompt Tokens | Completion Tokens | Total Tokens | Latency(s) | 결과/판정 | 비고 |")
    print("|------|------|------|---------------|-------------------|--------------|------------|----------|------|")

    # 노드 순서 및 메타데이터 정의
    node_order = ["관련성 체크", "쿼리 재작성", "retrieve", "check_constraints", "관련성 평가", "web_search", "제약 조건 경고", "답변 생성"]
    node_metadata = {
        "관련성 체크": {"step": "0", "desc": "레시피 관련성 체크", "timing_key": "check_relevance"},
        "쿼리 재작성": {"step": "1", "desc": "쿼리 재작성", "timing_key": "rewrite"},
        "retrieve": {"step": "2", "desc": "RAG 검색", "timing_key": "retrieve"},
        "check_constraints": {"step": "2.5", "desc": "제약 조건 체크", "timing_key": "check_constraints"},
        "관련성 평가": {"step": "3", "desc": "문서 관련성 평가", "timing_key": "grade"},
        "web_search": {"step": "4", "desc": "웹 검색", "timing_key": "web_search"},
        "제약 조건 경고": {"step": "5a", "desc": "제약 조건 경고", "timing_key": "generate"},
        "답변 생성": {"step": "5", "desc": "답변 생성", "timing_key": "generate"},
    }

    # 노드 순서대로 출력
    for node_name in node_order:
        tokens = _node_tokens.get(node_name, {"prompt": 0, "completion": 0, "total": 0})
        meta = node_metadata.get(node_name, {"step": "-", "desc": node_name, "timing_key": node_name})
        timing_ms = _node_timings.get(meta["timing_key"], 0)
        timing_sec = timing_ms / 1000 if timing_ms else 0

        if tokens["total"] > 0 or timing_sec > 0:
            prompt_str = str(tokens["prompt"]) if tokens["prompt"] > 0 else "-"
            completion_str = str(tokens["completion"]) if tokens["completion"] > 0 else "-"
            total_str = str(tokens["total"]) if tokens["total"] > 0 else "-"
            latency_str = f"{timing_sec:.1f}" if timing_sec > 0 else "-"
            print(f"| {meta['step']} | {node_name} | {meta['desc']} | {prompt_str} | {completion_str} | {total_str} | {latency_str} | - | - |")

    print("="*100 + "\n")

    # ✅ 2) 전체 합계 요약 표 (마크다운)
    print("="*100)
    print("📊 전체 합계 요약")
    print("="*100)
    print("| 구분 | Prompt Tokens | Completion Tokens | Total Tokens |")
    print("|------|---------------|-------------------|--------------|")
    print(f"| 합계 | {_token_accumulator['prompt']:,} | {_token_accumulator['completion']:,} | {_token_accumulator['total']:,} |")
    print("="*100 + "\n")

    # ✅ 3) 성능 병목 표: 시간 기준 랭킹 (마크다운)
    if _node_timings:
        print("="*100)
        print("⚡ 성능 병목 분석")
        print("="*100)
        print("| Rank | Node | Latency(s) | 비율 |")
        print("|------|------|------------|------|")

        sorted_timings = sorted(_node_timings.items(), key=lambda x: x[1], reverse=True)
        total_time = sum(_node_timings.values())

        for rank, (node_name, ms) in enumerate(sorted_timings, 1):
            sec = ms / 1000
            ratio = (ms / total_time * 100) if total_time > 0 else 0
            print(f"| {rank} | {node_name} | {sec:.1f} | ~{ratio:.0f}% |")

        print("="*100 + "\n")

    # 초기화
    _token_accumulator["prompt"] = 0
    _token_accumulator["completion"] = 0
    _token_accumulator["total"] = 0
    _node_tokens.clear()
