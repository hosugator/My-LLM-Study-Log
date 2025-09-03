# ─────────────────────────────────────────────────────────────────────────────
# [A] 42dot 1.3B: 단순 프롬프트 기반 생성
# ─────────────────────────────────────────────────────────────────────────────

# transformers 라이브러리 : Hugging Face에서 제공하는 NLP 모델 프레임워크
# 1. 모델 불러오기 → transformers.pipeline()에서 "text-generation" 작업과 model_id를 지정하여 자동으로 다운로드 및 로드.
# 2. 토크나이저 적용 → 입력 텍스트를 토큰(숫자 시퀀스)으로 변환.
# 3. 모델 추론 실행 → 입력 토큰을 모델에 넣고, 다음 토큰 예측을 반복하여 텍스트 생성.
# 4. 토큰 → 텍스트 변환 → 모델이 생성한 토큰을 사람이 읽을 수 있는 문자열로 복원.


# 1. 라이브러리 임포트
import transformers    
import torch

# 2. 사용할 모델 ID 지정
model_id = "42dot/42dot_LLM-SFT-1.3B"

# 3. 텍스트 생성 파이프라인 초기화
#    - task: "text-generation" (텍스트 생성)
#    - model_kwargs: float16로 로드(메모리 사용량 절감)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16}   # torch.float16 : 반정밀도(half-precision) 연산을 사용해 메모리 절약 & 속도 향상.
)

# 4. 모델을 평가(추론) 모드로 전환
pipeline.model.eval()

# 5. 단순 지시형 프롬프트 작성 (컨텍스트 없이 질문만)
prompt = "삼성전자에 대해 알려줘."

# 6. 파이프라인을 이용해 텍스트 생성 실행
#    - do_sample=True: 샘플링 활성화(다양성↑)
#    - temperature=0.3: 낮을수록 보수적/결정적
#    - top_p=0.9: nucleus sampling 임계값
outputs = pipeline(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    top_p=0.9
)

# 7. 반환 텍스트에서 프롬프트 부분을 제외하고 생성분만 출력
print(outputs[0]['generated_text'][len(prompt):])
