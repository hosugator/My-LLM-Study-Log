# hugging face model example code
# pip install torch transformers==4.40.0 accelerate

# ─────────────────────────────────────────────────────────────────────────────
# [B] Llama-3 Korean Bllossom 8B: 채팅 템플릿 기반 생성
# ─────────────────────────────────────────────────────────────────────────────

# 1. 라이브러리 임포트
import transformers
import torch

# 2. 사용할 모델 ID 지정 (한국어 특화 LoRA/튜닝 모델 예시) 
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"    

# 3. 텍스트 생성 파이프라인 초기화
#    - bfloat16 사용(최근 GPU에서 안정적)
#    - device_map="auto": 가용 디바이스에 자동 배치
pipeline = transformers.pipeline(     # Hugging Face Hub에서 다운로드 (transformers.pipeline)
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 4. 모델 평가(추론) 모드 전환
pipeline.model.eval()

# 5. 시스템 프롬프트(assistant 역할 규정)와 사용자 질문 준비
PROMPT = (
    "You are a helpful AI assistant. Please answer the user's questions kindly. "
    "당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."
)
instruction = "서울의 유명한 관광 코스를 만들어줄래?"

# 6. ChatGPT 형식의 대화 메시지 구성 (system/user 역할 구분)
messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
]

# 7. 토크나이저의 채팅 템플릿 적용 → 모델이 기대하는 입력 포맷으로 변환
#    - tokenize=False: 토크나이즈하지 않고 문자열 템플릿만 생성
#    - add_generation_prompt=True: assistant가 이어서 말하도록 프롬프트 앵커 추가
prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 8. 종료 토큰 설정 (모델이 어디서 멈출지 알려줌)
terminators = [
    pipeline.tokenizer.eos_token_id,                          # 시퀀스 종료 토큰
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")    # 대화 턴 종료 토큰
]

# 9. 텍스트 생성 실행 (길이, 샘플링 하이퍼파라미터 설정)
outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

# 10. 생성된 전체 텍스트에서 프롬프트를 제외하고 답변만 출력
print(outputs[0]["generated_text"][len(prompt):])
