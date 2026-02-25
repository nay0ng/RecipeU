"""
HCX API 토큰 계산 검증 스크립트
실제 API 응답 구조를 확인하고 토큰 추출이 정확한지 검증
"""
import os
from dotenv import load_dotenv
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage
import json

# 환경변수 로드
load_dotenv()

def test_hcx_token_response():
    """HCX API 응답 구조 확인"""
    print("="*80)
    print("HCX API 토큰 응답 구조 검증")
    print("="*80)

    # HCX LLM 생성
    llm = ChatClovaX(model="HCX-DASH-001", temperature=0.1, max_tokens=50)

    # 간단한 테스트 메시지
    test_message = "안녕하세요. 간단한 테스트 메시지입니다."

    print(f"\n📝 테스트 메시지: {test_message}\n")

    # LLM 호출
    response = llm.invoke([HumanMessage(content=test_message)])

    print("\n" + "="*80)
    print("1️⃣ 응답 객체의 전체 속성")
    print("="*80)
    print(f"응답 타입: {type(response)}")
    print(f"응답 속성 목록: {dir(response)}")

    print("\n" + "="*80)
    print("2️⃣ 응답 내용 (content)")
    print("="*80)
    print(f"응답 내용: {response.content}")

    print("\n" + "="*80)
    print("3️⃣ response_metadata 전체 구조")
    print("="*80)
    if hasattr(response, 'response_metadata'):
        print(json.dumps(response.response_metadata, indent=2, ensure_ascii=False))
    else:
        print("❌ response_metadata 속성 없음")

    print("\n" + "="*80)
    print("4️⃣ usage_metadata 전체 구조")
    print("="*80)
    if hasattr(response, 'usage_metadata'):
        print(json.dumps(response.usage_metadata, indent=2, ensure_ascii=False))
    else:
        print("❌ usage_metadata 속성 없음")

    print("\n" + "="*80)
    print("5️⃣ 토큰 추출 시도 (현재 코드 로직)")
    print("="*80)

    # 현재 코드의 토큰 추출 로직
    usage = None
    if hasattr(response, 'response_metadata'):
        usage = response.response_metadata.get('token_usage') or response.response_metadata.get('usage')
        print(f"✅ response_metadata에서 찾음: {usage}")
    elif hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        print(f"✅ usage_metadata에서 찾음: {usage}")

    if usage:
        print("\n토큰 필드 확인:")
        print(f"  - token_usage: {usage.get('token_usage')}")
        print(f"  - usage: {usage.get('usage')}")
        print(f"  - prompt_tokens: {usage.get('prompt_tokens')}")
        print(f"  - promptTokens: {usage.get('promptTokens')}")
        print(f"  - input_tokens: {usage.get('input_tokens')}")
        print(f"  - completion_tokens: {usage.get('completion_tokens')}")
        print(f"  - completionTokens: {usage.get('completionTokens')}")
        print(f"  - output_tokens: {usage.get('output_tokens')}")
        print(f"  - total_tokens: {usage.get('total_tokens')}")
        print(f"  - totalTokens: {usage.get('totalTokens')}")

        # 최종 추출값
        prompt_tokens = usage.get('prompt_tokens') or usage.get('promptTokens') or usage.get('input_tokens', 0)
        completion_tokens = usage.get('completion_tokens') or usage.get('completionTokens') or usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens') or usage.get('totalTokens', 0)

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        print("\n" + "="*80)
        print("6️⃣ 최종 추출된 토큰 (현재 코드 로직)")
        print("="*80)
        print(f"📥 입력 토큰 (prompt):     {prompt_tokens}")
        print(f"📤 출력 토큰 (completion): {completion_tokens}")
        print(f"📊 총 토큰 (total):        {total_tokens}")

        # 검증
        if total_tokens == prompt_tokens + completion_tokens:
            print("\n✅ 토큰 계산 일치: total = prompt + completion")
        else:
            print("\n⚠️  토큰 계산 불일치!")
            print(f"   계산값: {prompt_tokens + completion_tokens}")
            print(f"   API 반환값: {total_tokens}")
    else:
        print("\n❌ 토큰 정보를 찾을 수 없음!")

    print("\n" + "="*80)
    print("7️⃣ 원시 응답 객체 덤프")
    print("="*80)
    print(f"전체 응답: {response}")

    return response

if __name__ == "__main__":
    try:
        test_hcx_token_response()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
