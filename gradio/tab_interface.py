import gradio as gr

tab1 = gr.Interface(lambda x: x[::-1], inputs="text", outputs="text", title="Reverse")
tab2 = gr.Interface(
    lambda x: x.upper(), inputs="text", outputs="text", title="Uppercase"
)

gr.TabbedInterface([tab1, tab2], ["뒤집기", "대문자"]).launch()
