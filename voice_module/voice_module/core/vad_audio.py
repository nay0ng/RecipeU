# import pyaudio
import wave
import torch
import numpy as np
import time
from collections import deque

from pathlib import Path

class VADAudioRecorder:
    def __init__(self):
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # 기본 VAD 설정: 반응 속도를 위해 0.5초로 짧게 잡음
        # self.SILENCE_THRESHOLD = 0.5 
        self.START_THRESHOLD = 0.6   # 말 시작 감지 (더 높게)
        self.END_THRESHOLD   = 0.45  # 말 끝/침묵 판정 (더 낮게)
        self.SILENCE_DURATION = 0.5  
        
        # [New] 외부에서 설정 가능한 대기 타임아웃 (초 단위)
        # 이 시간이 지날 때까지 말을 안 하면 None을 반환함
        self.max_listen_timeout = None 
        self.last_listen_time = time.time()

        print("VAD 모델 로딩 중...")
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           trust_repo=True)
        (_, _, _, _, _) = utils 
        print("VAD 모델 준비 완료")
        
        self.p = pyaudio.PyAudio()

    def listen_and_record(self, stop_event=None, out_dir="."):
        stream = self.p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK)

        print("\n🎤 Listening... (말씀해 보세요)")
        
        audio_buffer = []
        is_recording = False
        silence_start_time = None
        self.last_listen_time = time.time() # 대기 시작 시간 초기화
        
        pre_roll_buffer = deque(maxlen=20)

        while True:
            try:
                # ✅ Streamlit Stop 버튼에서 멈추게 하는 핵심
                if stop_event is not None and stop_event.is_set():
                    break
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                tensor_audio = torch.from_numpy(audio_float32)
                
                speech_prob = self.model(tensor_audio, self.RATE).item()

                # --- 1. 대기 상태 (Listening) ---
                if not is_recording:
                    pre_roll_buffer.append(data)
                    
                    # (A) 말 시작 감지
                    if speech_prob > self.START_THRESHOLD:
                        print("\n🔴 감지됨! 녹음 시작...")
                        is_recording = True
                        audio_buffer.extend(pre_roll_buffer)
                        silence_start_time = None
                        self.max_listen_timeout = None # 말 시작했으니 타임아웃 해제

                    # (B) 타임아웃 체크 (말 안 하고 버티는 경우)
                    elif self.max_listen_timeout is not None:
                        elapsed = time.time() - self.last_listen_time
                        if elapsed > self.max_listen_timeout:
                            print(f"\n대기 시간 초과 ({self.max_listen_timeout}s)... 강제 전송")
                            yield None # 타임아웃 신호(None) 전송
                            self.max_listen_timeout = None # 리셋
                            self.last_listen_time = time.time()

                # --- 2. 녹음 상태 (Recording) ---
                else:
                    audio_buffer.append(data)
                    
                    if speech_prob > self.END_THRESHOLD:
                        silence_start_time = None
                    else:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        
                        # 침묵이 0.5초 지속되면 녹음 종료
                        elif time.time() - silence_start_time > self.SILENCE_DURATION:
                            print("⏹️ 1차 녹음 종료 (분석 시작)")
                            
                            filename = f"voice_input_{int(time.time())}.wav"
                            # self._save_wav(filename, audio_buffer)
                            self._save_wav(str(filename), audio_buffer)
                            
                            # yield filename # 파일명 전송
                            yield str(filename)
                            
                            # 초기화 및 다시 대기
                            audio_buffer = []
                            is_recording = False
                            pre_roll_buffer.clear()
                            self.last_listen_time = time.time() # 대기 타이머 리셋
                            print("\n🎤 Listening...")

            except KeyboardInterrupt:
                break
        
        stream.stop_stream()
        stream.close()

    def _save_wav(self, filename, frames):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def close(self):
        self.p.terminate()