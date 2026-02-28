"""
visualize_graph.py
LangGraph 에이전트 구조를 시각화하는 스크립트

실행:
    cd backend
    python visualize_graph.py

출력 파일:
    graph.png      - PNG 이미지 (Mermaid/Graphviz 필요)
    graph.mmd      - Mermaid 다이어그램 텍스트 (항상 생성)
    (콘솔)         - ASCII 다이어그램 (항상 출력)

Mermaid 뷰어: https://mermaid.live
"""

import sys
import os

# backend 경로를 sys.path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from features.chat.agent import create_chat_agent

def main():
    print("그래프 생성 중 (rag_system=None)...")
    compiled = create_chat_agent(rag_system=None)
    graph = compiled.get_graph()

    # ── 1. ASCII 출력 (항상 동작) ──────────────────────────
    print("\n" + "="*60)
    print("ASCII 다이어그램")
    print("="*60)
    graph.print_ascii()

    # ── 2. Mermaid 텍스트 저장 (항상 동작) ─────────────────
    mermaid_text = graph.draw_mermaid()
    output_dir = os.path.dirname(__file__)
    mmd_path = os.path.join(output_dir, "graph.mmd")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_text)
    print(f"\n[OK] Mermaid 파일 저장: {mmd_path}")
    print("     → https://mermaid.live 에 붙여넣기하면 시각화 가능\n")
    print(mermaid_text)

    # ── 3. PNG 저장 시도 ────────────────────────────────────
    png_path = os.path.join(output_dir, "graph.png")

    # 방법 A: draw_mermaid_png (playwright 또는 pyppeteer 필요)
    try:
        png_data = graph.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"[OK] PNG 저장 (Mermaid): {png_path}")
        return
    except Exception as e:
        print(f"[SKIP] Mermaid PNG 실패: {e}")

    # 방법 B: draw_png (graphviz 필요)
    try:
        png_data = graph.draw_png()
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"[OK] PNG 저장 (Graphviz): {png_path}")
        return
    except Exception as e:
        print(f"[SKIP] Graphviz PNG 실패: {e}")

    print("[INFO] PNG 생성 불가 - graph.mmd 파일을 mermaid.live에서 확인하세요")


if __name__ == "__main__":
    main()
