# debug_tools.py
import json

with open("../recipe_data/recipes_180000_with_tools.json", "r") as f:
    recipes = json.load(f)

# cooking_tools 있는 레시피 찾기
has_tools = [r for r in recipes if r.get("cooking_tools")]
print(f"도구 있는 레시피: {len(has_tools)}개")
print(f"샘플: {has_tools[0]['title']} → {has_tools[0]['cooking_tools']}")

# tool_buf 시뮬레이션
tool_buf = []
for recipe in recipes[:100]:
    for tool in recipe.get("cooking_tools", []):
        if tool:
            tool_buf.append({
                "recipe_id": recipe["recipe_id"],
                "tool_name": tool
            })
print(f"\n첫 100개 레시피에서 tool_buf: {len(tool_buf)}개")
print(f"샘플: {tool_buf[:3]}")