# 멀티턴 대화형 챗봇 웹 애플리케이션
# 1. 라이브러리 및 API 키 설정
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # .env에서 API 키 읽기

client = OpenAI(api_key=api_key)

# 2. GPT 응답 생성 함수 정의
def chat_with_bot(messages):
    gen = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        stream=True
    )
    return gen

# 3. 최대 턴 수 설정 (3턴 = 총 6개의 메시지: user 3 + assistant 3)
MAX_TURNS = 3

# 4. Streamlit 앱 구성
def main():
    st.title("Multi-turn Chatbot (최근 3턴 유지)")
    st.chat_input(placeholder="대화를 입력해주세요.", key="chat_input")

    # 5. 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 6. 기존 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 7. 사용자 입력 처리
    if user_input := st.session_state["chat_input"]:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 8. 메시지가 너무 길면 최근 3턴만 남기기 (총 6개 메시지)
        if len(st.session_state.messages) > MAX_TURNS * 2:
            st.session_state.messages = st.session_state.messages[-MAX_TURNS * 2:]

        # 9. GPT 응답 생성
        gen = chat_with_bot(st.session_state.messages)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in gen:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # 다시 한 번 길이 제한 체크
            if len(st.session_state.messages) > MAX_TURNS * 2:
                st.session_state.messages = st.session_state.messages[-MAX_TURNS * 2:]

# 10. 앱 실행
if __name__ == "__main__":
    main()

