# Single-turn : Interface 사용 - 대화가 누적되지 않고 매번 랜덤 답변만 출력

# 1) 라이브러리 임포트 및 환경 변수 로드
import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2) 응답 생성 함수 (단순 반환: 스트리밍 없음)
def respond(message: str):
    if not message or not message.strip():
        return "질문을 입력해 주세요."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "간결하고 친절한 한국어 비서처럼 답하세요."},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=600,
            stream=False,  # ✅ 스트리밍 비활성화
        )
        return resp.choices[0].message.content  # ✅ 문자열 한 번에 반환
    except Exception as e:
        return f"[오류] {type(e).__name__}: {e}"

# 3) Gradio 인터페이스 (Single-turn)
demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(placeholder="질문을 입력하세요"),
    outputs="text",
    title="GPT 싱글턴 (비스트리밍)",
)

# 4) 실행 (queue 불필요)
demo.launch(share=False)
