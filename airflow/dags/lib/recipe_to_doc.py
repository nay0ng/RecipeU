from langchain_core.documents import Document


def recipe_to_document(recipe: dict) -> Document:
    ingredients_text = ", ".join(
        f"{ing.get('name', '')} {ing.get('amount', '')}".strip()
        for ing in recipe.get("ingredients", [])
    )

    steps_text = "\n".join(recipe.get("steps", []))

    recipe_text = f"""제목: {recipe.get('title', '')}

                소개: {recipe.get('intro', '')}

                재료: {ingredients_text}

                조리법:
                {steps_text}

                조리시간: {recipe.get('cook_time', 'unknown')}
                난이도: {recipe.get('level', 'unknown')}
                분량: {recipe.get('portion', 'unknown')}
            """

    metadata = {
        "recipe_id": str(recipe.get("recipe_id") or ""),
        "title": recipe.get("title") or "",
        "level": recipe.get("level") or "",
        "cook_time": recipe.get("cook_time") or "unknown",
        "source": recipe.get("detail_url") or "",
    }

    return Document(
        page_content=recipe_text.strip(),
        metadata=metadata,  # ❗️ 절대 필터링 생각하지 마
    )
