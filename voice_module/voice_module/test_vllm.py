from openai import OpenAI

# 1. 클라이언트 설정 (중요!)
client = OpenAI(
    base_url="http://localhost:8080/v1",  # vLLM 서버 주소 (뒤에 /v1 꼭 붙여야 함)
    api_key="vllm",                       # vLLM은 키가 필요 없지만, 공란이면 에러나니 아무거나 넣음
)

# 2. 모델 이름 설정 (서버에 띄운 이름과 똑같아야 함)
model_id = "Qwen/Qwen3-4B-Instruct-2507"

# 3. 요청 보내기
response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "너는 친절한 AI 어시스턴트야."},
        {"role": "user", "content": "양지머리로 만드는 레시피를 알려줘"}
    ],
    temperature=0.7,
)

# 4. 결과 출력
print(response.choices[0].message.content)