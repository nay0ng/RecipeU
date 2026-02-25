#!/bin/bash
# chmod +x start_servers.sh 로 permission 허용
# ./start_servers.sh 로 실행

# ---------------------------------------------------------
# [0] 환경 변수 설정 (경로 수정 필수!!!)
# ---------------------------------------------------------
PROJECT_DIR="/workspace/voice_module"
VENV_MAIN="/workspace/venv"
VENV_LLM="/workspace/venv_llm"
SESSION="voice_project"

# 마우스 사용 켜기
echo "set -g mouse on" > ~/.tmux.conf


# ---------------------------------------------------------
# [0.5] 필수 도구 설치 (tmux, psmisc)
# ---------------------------------------------------------
# psmisc는 포트 죽이는 명령어(fuser)를 쓰기 위해 필요함
if ! command -v fuser &> /dev/null || ! command -v tmux &> /dev/null
then
    echo "필수 도구(tmux, psmisc)가 없습니다. 설치합니다..."
    apt-get update && apt-get install -y tmux psmisc
    echo "설치 완료!"
fi

# ---------------------------------------------------------
# [1] 기존 좀비 프로세스 사살 (가장 중요!!)
# ---------------------------------------------------------
echo "🧹 기존에 열린 포트 청소 중..."
fuser -k 5000/tcp  2>/dev/null
fuser -k 8011/tcp  2>/dev/null
fuser -k 8012/tcp  2>/dev/null
fuser -k 8013/tcp  2>/dev/null
echo "✨ 포트 청소 완료! 깨끗한 상태에서 시작합니다."

# ---------------------------------------------------------
# [2] 세션 시작
# ---------------------------------------------------------
# 기존 세션 종료
tmux kill-session -t $SESSION 2>/dev/null

# 새 세션 시작
tmux new-session -d -s $SESSION

# ---------------------------------------------------------
# [Pane 1] 우측 상단: vLLM (venv_llm)
# ---------------------------------------------------------
tmux rename-window 'servers'
tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION "source $VENV_LLM/bin/activate" C-m
tmux send-keys -t $SESSION 'vllm serve jjjunho/Qwen3-4B-Instruct-2507-Korean-AWQ --port 5000 --gpu-memory-utilization 0.6 --max-model-len 4096' C-m

# ---------------------------------------------------------
# [Pane 2] 좌측: STT (venv)
# ---------------------------------------------------------
tmux split-window -h
tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION "source $VENV_MAIN/bin/activate" C-m
tmux send-keys -t $SESSION 'python servers/stt_server.py' C-m

# ---------------------------------------------------------
# [Pane 3] 우측 하단: TTS (venv)
# ---------------------------------------------------------
tmux select-pane -t 0
tmux split-window -v
tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION "source $VENV_MAIN/bin/activate" C-m
tmux send-keys -t $SESSION 'python servers/tts_server.py' C-m

# ---------------------------------------------------------
# [Pane 4] 좌측 하단: LLM Client (venv_llm)
# ---------------------------------------------------------
tmux select-pane -t 2
tmux split-window -v
tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION "source $VENV_LLM/bin/activate" C-m
tmux send-keys -t $SESSION 'python servers/llm_server.py' C-m

# ---------------------------------------------------------
# 마무리
# ---------------------------------------------------------
tmux select-layout tiled
tmux attach -t $SESSION