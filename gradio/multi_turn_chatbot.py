# Multi-turn : ChatInterface 사용
import gradio as gr
import random
import time


# 멀티턴 챗봇 함수
def respond(message, history):
    response = random.choice(["안녕하세요", "사랑합니다", "배가 고파요"])
    streamed = ""
    for char in response:
        streamed += char
        time.sleep(0.05)  # 스트리밍 흉내내기
        yield streamed  # ChatInterface가 자동으로 history에 누적


# ChatInterface 생성
demo = gr.ChatInterface(
    fn=respond,
    title="스트리밍 멀티턴 챗봇",
    description="간단한 멀티턴 대화 예시",
)

demo.queue()  # 스트리밍을 위해 queue() 필요
demo.launch()
