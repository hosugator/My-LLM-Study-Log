import gradio as gr

def greet(name, is_morning, temperature):
    try:
        salutation = "좋은 아침입니다" if is_morning else "좋은 저녁입니다"
        greeting = f"{salutation} {name}씨. 오늘은 섭씨 {temperature}도입니다."
        fahrenheit = (temperature * 9 / 5) + 32
        return greeting, f"화씨는 {fahrenheit:.1f}℉ 입니다"
    except Exception as e:
        return "오류 발생", f"에러: {str(e)}"

demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "text"],
)

demo.launch(share=True)  # Gradio 앱을 외부에 공유 가능한 공개 링크로 실행