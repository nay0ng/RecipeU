#!/bin/bash
# 파일명: 2_go_others.sh
# 실행법: ./2_go_others.sh (새 터미널에서 실행)

SESSION="voice_project"

echo "🚀 나머지 서버들을 가동합니다!"

# [Pane 2] STT 실행 (포트 8011)
# tmux send-keys -t $SESSION:0.2 'python servers/stt_server.py' C-m

# [Pane 3] TTS 실행 (포트 8012)
tmux send-keys -t $SESSION:0.1 'python servers/tts_server.py' C-m
# (참고: tmux 레이아웃에 따라 번호가 다를 수 있어서 안전하게 순서대로 보냅니다)
# 화면상 위치: 우측 하단

# [Pane 4] Client 실행 (포트 8013)
tmux send-keys -t $SESSION:0.3 'python servers/llm_server.py' C-m

echo "✅ 실행 명령 전송 완료! tmux 화면을 확인하세요."