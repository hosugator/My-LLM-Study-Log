# Single-turn : Interface 사용 - 대화가 누적되지 않고 매번 랜덤 답변만 출력

import gradio as gr
import random
import time

def respond(message):
    response = random.choice(["안녕하세요", "사랑합니다", "배가 고파요"])
    streamed = ""
    for ch in response:
        streamed += ch
        time.sleep(0.05)  # 스트리밍 효과
        yield streamed     # 문자열만 연속 yield → 단일 출력 필드에 스트리밍 표시

demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(placeholder="메시지를 입력하세요"),
    outputs="text",
    title="싱글턴 스트리밍 예시"
)

demo.queue()   # 스트리밍을 위해 필요
demo.launch()
