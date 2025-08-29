# 1. 라이브러리 및 API 키 설정
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 OpenAI API 키 불러오기

client = OpenAI()

# 2. GPT 모델로 단일 응답을 생성하는 함수 정의
def chat_single_turn(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",              # 사용할 모델
        messages=[                        # 단일 메시지 (기억 없음)
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,                  # 창의성
        max_tokens=1024                   # 응답 최대 토큰 수
    )
    return response.choices[0].message.content  # 첫 번째 응답 반환

# 3. Streamlit 앱 UI 구성
def main():
    st.title("Single-turn Chatbot with Streamlit and OpenAI")  # 앱 타이틀

    # 4. 사용자 입력 받기
    prompt = st.text_input("질문을 입력하세요:")  # 단일 입력창

    # 5. 버튼 클릭 시 응답 출력
    if st.button("질문하기") and prompt:
        with st.spinner("GPT가 답변 중입니다..."):
            response = chat_single_turn(prompt)  # 단일 응답 호출
            st.markdown("### GPT의 답변:")
            st.write(response)

# 앱 실행
if __name__ == "__main__":
    main()
