def print_recipe_token_summary():
    """레시피 생성 토큰 사용량 요약 출력 (마크다운 형식)"""
    if _token_accumulator["total"] == 0:
        return

    print(f"\n{'🔷'*30}")
    print(f"{'  '*10}📊 레시피 생성 토큰 사용량 요약")
    print(f"{'🔷'*30}")
    print(f"📥 총 입력 토큰 (prompt):     {_token_accumulator['prompt']:,} tokens")
    print(f"📤 총 출력 토큰 (completion): {_token_accumulator['completion']:,} tokens")
    print(f"📊 총합 (total):              {_token_accumulator['total']:,} tokens")
    print(f"{'🔷'*30}\n")

    # ✅ 1) 단계별 토큰 요약 표 (마크다운)
    print("\n" + "="*100)
    print("- 📋 단계별 상세 요약\n")
    print("| Step | 설명 | Prompt Tokens | Completion Tokens | Total Tokens |")
    print("|------|------|---------------|-------------------|--------------|")

    # 단계 순서 정의
    step_order = ["검색 쿼리 추출", "레시피 생성"]
    step_metadata = {
        "검색 쿼리 추출": {"step": "1", "desc": "검색 쿼리 추출"},
        "레시피 생성": {"step": "2", "desc": "레시피 생성"},
    }

    # 단계 순서대로 출력
    for step_name in step_order:
        tokens = _step_tokens.get(step_name, {"prompt": 0, "completion": 0, "total": 0})
        meta = step_metadata.get(step_name, {"step": "-", "desc": step_name})

        if tokens["total"] > 0:
            prompt_str = str(tokens["prompt"]) if tokens["prompt"] > 0 else "-"
            completion_str = str(tokens["completion"]) if tokens["completion"] > 0 else "-"
            total_str = str(tokens["total"]) if tokens["total"] > 0 else "-"
            print(f"| {meta['step']} | {meta['desc']} | {prompt_str} | {completion_str} | {total_str} |")

    # ✅ 2) 전체 합계 요약 표 (마크다운)
    print("\n- 📊 전체 합계 요약\n")
    print("| 구분 | Prompt Tokens | Completion Tokens | Total Tokens |")
    print("|------|---------------|-------------------|--------------|")
    print(f"| 합계 | {_token_accumulator['prompt']:,} | {_token_accumulator['completion']:,} | {_token_accumulator['total']:,} |")

    print("="*100 + "\n")

    # 초기화
    _token_accumulator["prompt"] = 0
    _token_accumulator["completion"] = 0
    _token_accumulator["total"] = 0
    _step_tokens.clear()
