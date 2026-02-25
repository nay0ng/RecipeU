from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="vllm",
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[{"role": "user", "content": "김치찌개 만드는 레시피 알려줘"}],
    stream=True,  # 스트리밍 활성화
)

print("답변: ", end="", flush=True)

# 한 글자(청크)씩 받아서 바로 출력
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print() # 줄바꿈
