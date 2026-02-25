#!/bin/bash
# 파일명: 1_ready_vllm.sh
# 실행법: ./1_ready_vllm.sh
# chmod +x 1_ready_vllm.sh 2_go_others.sh


# psmisc는 포트 죽이는 명령어(fuser)를 쓰기 위해 필요함
if ! command -v fuser &> /dev/null || ! command -v tmux &> /dev/null
then
    echo "필수 도구(tmux, psmisc)가 없습니다. 설치합니다..."
    apt-get update && apt-get install -y tmux psmisc
    echo "설치 완료!"
fi

# --- [1] 변수 설정 ---
PROJECT_DIR="/workspace/voice_module"
VENV_MAIN="/workspace/venv"
VENV_LLM="/workspace/venv_llm"
SESSION="voice_project"

# 마우스 사용 켜기
echo "set -g mouse on" > ~/.tmux.conf

# ========================================================
# [추가됨] python-dotenv 설치 확인 (환경변수 로드용)
# ========================================================
echo "python-dotenv 설치 확인 중..."
if ! $VENV_MAIN/bin/python -c "import dotenv" &> /dev/null; then
    echo "❌ python-dotenv가 없습니다. 설치를 진행합니다..."
    $VENV_MAIN/bin/pip install python-dotenv
    echo "✅ python-dotenv 설치 완료!"
else
    echo "python-dotenv가 이미 설치되어 있습니다."
fi

# ========================================================
# [추가됨] NLTK 데이터 자동 다운로드 (에러 방지용)
# TTS 서버가 사용하는 가상환경(VENV_MAIN)을 이용하여 다운로드
# ========================================================
echo "📦 NLTK 필수 데이터 확인 및 다운로드 중..."
$VENV_MAIN/bin/python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); print('✅ NLTK averaged_perceptron_tagger_eng 준비 완료')"


# --- [2] 청소 (대학살) ---
echo "🧹 기존 프로세스 정리 중..."
pkill -9 python  # 모든 파이썬 종료
fuser -k 5000/tcp 2>/dev/null
# fuser -k 8011/tcp 2>/dev/null
fuser -k 8012/tcp 2>/dev/null
fuser -k 8013/tcp 2>/dev/null

# --- [3] tmux 세션 시작 ---
tmux kill-session -t $SESSION 2>/dev/null
tmux new-session -d -s $SESSION

# --- [4] 화면 분할 및 vLLM 실행 ---

# [Pane 1: 우측 상단] vLLM (주인공) - 바로 실행!
tmux rename-window 'servers'
tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION "source $VENV_LLM/bin/activate" C-m
# 메모리 점유율 0.4로 설정 (안전하게)
# tmux send-keys -t $SESSION 'vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 5000 --gpu-memory-utilization 0.4 --max-model-len 4096' C-m
# tmux send-keys -t $SESSION 'vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 5000 --gpu-memory-utilization 0.8 --max-model-len 4096' C-m
# tmux send-keys -t $SESSION 'vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 5000 --gpu-memory-utilization 0.4 --max-model-len 1024' C-m
# tmux send-keys -t $SESSION 'vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 5000 --gpu-memory-utilization 0.6 --max-model-len 2048' C-m
tmux send-keys -t $SESSION 'vllm serve jjjunho/Qwen3-4B-Instruct-2507-Korean-AWQ --port 5000 --gpu-memory-utilization 0.6 --max-model-len 4096' C-m

# [Pane 2: 좌측] STT (대기)
tmux split-window -h
# tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
# tmux send-keys -t $SESSION "source $VENV_MAIN/bin/activate" C-m
# tmux send-keys -t $SESSION "echo '💤 vLLM 로딩 끝나면 2번 스크립트를 실행하세요...'" C-m

# [Pane 3: 우측 하단] TTS (대기)
tmux select-pane -t 0
tmux split-window -v
tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION "source $VENV_MAIN/bin/activate" C-m
tmux send-keys -t $SESSION "echo '💤 대기 중...'" C-m

# [Pane 4: 좌측 하단] Client (대기)
tmux select-pane -t 2
tmux split-window -v
tmux send-keys -t $SESSION "cd $PROJECT_DIR" C-m
tmux send-keys -t $SESSION "source $VENV_LLM/bin/activate" C-m
tmux send-keys -t $SESSION "echo '💤 대기 중...'" C-m

# --- [5] 접속 ---
tmux select-layout tiled
tmux attach -t $SESSION