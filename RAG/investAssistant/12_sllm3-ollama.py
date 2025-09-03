# ─────────────────────────────────────────────────────────────
# Ollama: gemma3:4b 모델 사용 (chat API)
# ─────────────────────────────────────────────────────────────
# 사전 준비:
#   1) Ollama 설치 및 실행 (로컬에서 ollama serve) : https://ollama.com/download OllamaSetup.exe 실행
#   2) 모델 다운로드:  ollama pull gemma3:4b
#   3) pip install ollama => 0.5.3

# 1. 라이브러리 임포트 -----------------------------------------------------------
import ollama

# 2. 사용할 모델 이름 지정 -------------------------------------------------------
MODEL_NAME = "gemma3:4b"

# 3. 시스템/사용자 메시지 정의 (Chat 형식) ----------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Please answer the user's questions kindly. "
    "당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 친절하게 답변하세요."
)
USER_INSTRUCTION = "서울의 유명한 관광 코스를 만들어줄래?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_INSTRUCTION},
]

# 4. 생성 옵션 설정 --------------------------------------------------------------
#    - num_predict: 생성 토큰 수 (HuggingFace의 max_new_tokens 대응)
#    - temperature, top_p: 샘플링 하이퍼파라미터
#    - stop: 필요 시 종료 토큰/문구 지정 가능 (예: ["</s>"])
options = {
    "num_predict": 512,
    "temperature": 0.6,
    "top_p": 0.9,
    # "stop": ["</s>"],  # 필요 시 사용
}

# 5. Ollama 서버에 요청 보내기 ---------------------------------------------------
#    - stream=False: 전체 응답을 한 번에 수신
response = ollama.chat(
    model=MODEL_NAME,
    messages=messages,
    options=options,
    stream=False,
)

# 6. 응답 출력 -------------------------------------------------------------------
print(response["message"]["content"])
