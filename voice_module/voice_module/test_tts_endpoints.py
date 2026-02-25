"""
TTS 엔드포인트 비교 테스트
- /synthesize: 전체 오디오를 한 번에 반환 (WAV)
- /synthesize/stream: 청크 단위로 스트리밍 반환 (Raw PCM)

실행:
    python test_tts_endpoints.py
"""

import requests
import time
import wave
import io
import os
from datetime import datetime

# ============================================================================
# 설정
# ============================================================================

TTS_SERVER_URL = "http://localhost:8012"  # RunPod URL로 변경 가능
# TTS_SERVER_URL = "http://<runpod-ip>:8012"

TEST_TEXTS = [
    "안녕하세요.",  # 짧은 문장
    "오늘의 레시피는 맛있는 김치찌개입니다.",  # 중간 문장
    "먼저 냄비에 물을 붓고 끓인 다음, 김치와 돼지고기를 넣어주세요. 그 다음 양파와 두부를 넣고 5분간 더 끓여주시면 됩니다.",  # 긴 문장
]

TONE = "kiwi"
OUTPUT_DIR = "./test_outputs"


# ============================================================================
# 테스트 함수
# ============================================================================

def test_synthesize(text: str, save_file: bool = True) -> dict:
    """
    /synthesize 엔드포인트 테스트 (전체 WAV 반환)
    """
    url = f"{TTS_SERVER_URL}/synthesize"
    payload = {
        "text": text,
        "tone": TONE,
        "text_lang": "ko",
        "speed_factor": 1.0
    }

    start_time = time.time()
    first_byte_time = None

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        audio_chunks = []
        for chunk in response.iter_content(chunk_size=4096):
            if first_byte_time is None:
                first_byte_time = time.time()
            audio_chunks.append(chunk)

        total_time = time.time() - start_time
        ttfb = first_byte_time - start_time if first_byte_time else total_time

        audio_data = b''.join(audio_chunks)

        # 파일 저장
        if save_file:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S")
            filepath = os.path.join(OUTPUT_DIR, f"synthesize_{timestamp}.wav")
            with open(filepath, 'wb') as f:
                f.write(audio_data)
        else:
            filepath = None

        return {
            "success": True,
            "endpoint": "/synthesize",
            "text_length": len(text),
            "audio_size_bytes": len(audio_data),
            "ttfb_ms": round(ttfb * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "filepath": filepath
        }

    except Exception as e:
        return {
            "success": False,
            "endpoint": "/synthesize",
            "error": str(e)
        }


def test_synthesize_stream(text: str, save_file: bool = True) -> dict:
    """
    /synthesize/stream 엔드포인트 테스트 (PCM 스트리밍)
    """
    url = f"{TTS_SERVER_URL}/synthesize/stream"
    payload = {
        "text": text,
        "tone": TONE,
        "text_lang": "ko",
        "speed_factor": 1.0
    }

    start_time = time.time()
    first_byte_time = None
    chunk_count = 0

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        # 헤더에서 오디오 정보 가져오기
        sample_rate = int(response.headers.get("X-Sample-Rate", 32000))
        channels = int(response.headers.get("X-Channels", 1))
        sample_width = int(response.headers.get("X-Sample-Width", 2))

        audio_chunks = []
        for chunk in response.iter_content(chunk_size=4096):
            if first_byte_time is None:
                first_byte_time = time.time()
            audio_chunks.append(chunk)
            chunk_count += 1

        total_time = time.time() - start_time
        ttfb = first_byte_time - start_time if first_byte_time else total_time

        pcm_data = b''.join(audio_chunks)

        # PCM -> WAV 변환 후 저장
        if save_file and pcm_data:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S")
            filepath = os.path.join(OUTPUT_DIR, f"stream_{timestamp}.wav")

            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
        else:
            filepath = None

        return {
            "success": True,
            "endpoint": "/synthesize/stream",
            "text_length": len(text),
            "audio_size_bytes": len(pcm_data),
            "chunk_count": chunk_count,
            "sample_rate": sample_rate,
            "ttfb_ms": round(ttfb * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "filepath": filepath
        }

    except Exception as e:
        return {
            "success": False,
            "endpoint": "/synthesize/stream",
            "error": str(e)
        }


def health_check() -> bool:
    """서버 상태 확인"""
    try:
        response = requests.get(f"{TTS_SERVER_URL}/health", timeout=5)
        data = response.json()
        print(f"서버 상태: {data.get('status')}")
        print(f"사용 가능한 레퍼런스: {data.get('available_references')}개")
        return data.get('status') == 'healthy'
    except Exception as e:
        print(f"서버 연결 실패: {e}")
        return False


def run_comparison_test():
    """두 엔드포인트 비교 테스트 실행"""
    print("=" * 70)
    print("TTS 엔드포인트 비교 테스트")
    print("=" * 70)
    print(f"서버: {TTS_SERVER_URL}")
    print(f"톤: {TONE}")
    print()

    # 서버 상태 확인
    if not health_check():
        print("\n서버에 연결할 수 없습니다. 서버 URL을 확인해주세요.")
        return

    print()
    print("-" * 70)

    results = []

    for i, text in enumerate(TEST_TEXTS, 1):
        print(f"\n[테스트 {i}] 텍스트 길이: {len(text)}자")
        print(f"내용: {text[:50]}{'...' if len(text) > 50 else ''}")
        print()

        # /synthesize 테스트
        print("  /synthesize 테스트 중...")
        result1 = test_synthesize(text)
        if result1["success"]:
            print(f"    TTFB: {result1['ttfb_ms']}ms")
            print(f"    총 시간: {result1['total_time_ms']}ms")
            print(f"    파일 크기: {result1['audio_size_bytes']:,} bytes")
        else:
            print(f"    실패: {result1.get('error')}")

        # /synthesize/stream 테스트
        print("  /synthesize/stream 테스트 중...")
        result2 = test_synthesize_stream(text)
        if result2["success"]:
            print(f"    TTFB: {result2['ttfb_ms']}ms")
            print(f"    총 시간: {result2['total_time_ms']}ms")
            print(f"    청크 수: {result2['chunk_count']}")
            print(f"    파일 크기: {result2['audio_size_bytes']:,} bytes")
        else:
            print(f"    실패: {result2.get('error')}")

        # 비교
        if result1["success"] and result2["success"]:
            ttfb_diff = result1["ttfb_ms"] - result2["ttfb_ms"]
            total_diff = result1["total_time_ms"] - result2["total_time_ms"]
            print()
            print(f"  [비교 결과]")
            print(f"    TTFB 차이: {ttfb_diff:+.2f}ms (양수면 stream이 빠름)")
            print(f"    총 시간 차이: {total_diff:+.2f}ms (양수면 stream이 빠름)")

            results.append({
                "text_length": len(text),
                "synthesize_ttfb": result1["ttfb_ms"],
                "stream_ttfb": result2["ttfb_ms"],
                "synthesize_total": result1["total_time_ms"],
                "stream_total": result2["total_time_ms"],
            })

        print("-" * 70)

    # 최종 요약
    if results:
        print("\n" + "=" * 70)
        print("최종 요약")
        print("=" * 70)
        print()
        print(f"{'텍스트 길이':<12} {'synthesize TTFB':<18} {'stream TTFB':<15} {'승자':<10}")
        print("-" * 60)

        for r in results:
            winner = "stream" if r["stream_ttfb"] < r["synthesize_ttfb"] else "synthesize"
            print(f"{r['text_length']:<12} {r['synthesize_ttfb']:<18.2f} {r['stream_ttfb']:<15.2f} {winner:<10}")

        print()
        print(f"출력 파일 위치: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    run_comparison_test()
