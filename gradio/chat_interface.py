import gradio as gr

def respond(message, history):
    return "안녕하세요!"
    #return "안녕하세요!", history + [[message, "안녕하세요!"]] # history는 대화 내역이며, 새로운 message와 응답을 [[질문, 답변]] 형태로 붙여주는 방식

gr.ChatInterface(respond).launch()