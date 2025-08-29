import gradio as gr

with gr.Blocks() as demo:
    txt = gr.Textbox(label="이름을 입력하세요")
    btn = gr.Button('출력')
    out = gr.Textbox(label="출력")
    
    def echo(x):
        return f"안녕하세요, {x}님!"
    
    #btn.click(echo, inputs=txt, outputs=out)
    btn.click(echo, txt, out)

demo.launch(share=True)