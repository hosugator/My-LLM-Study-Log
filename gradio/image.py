import numpy as np
import gradio as gr

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

# gr.Image는 이미지만 전문으로 입력 또는 출력하는 컴포넌트
demo = gr.Interface(sepia, gr.Image(width=200, height=200), "image")
demo.launch(server_name='0.0.0.0')

