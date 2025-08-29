# 멀티턴 대화형 챗봇 웹 애플리케이션
# 1. 라이브러리 및 API 키 설정
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 OpenAI API 키 불러오기

client = OpenAI()

# 2. OpenAI GPT와 대화하는 함수 정의
def chat_with_bot(messages):
    gen = client.chat.completions.create(
        model="gpt-4o-mini",            # 사용할 GPT 모델 (gpt-4o-mini)
        messages=messages,              # 지금까지의 대화 내용 전달
        temperature=0.7,                # 창의성 조절 (0.0 ~ 1.0)
        max_tokens=4096,                # 최대 토큰 수 제한
        stream=True                     # 실시간 스트리밍 응답 활성화
    )
    return gen                          # 생성된 응답 스트림 반환

# 3. Streamlit 앱 구성
def main():
    st.title("Multi-turn Chatbot with Streamlit and OpenAI")  # 앱 제목
    st.chat_input(placeholder="대화를 입력해주세요.", key="chat_input")  # 입력창 생성

    # 4. 세션 상태에 메시지 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []  # 대화 이력이 없으면 빈 리스트로 초기화

    # 5. 이전 대화 내용 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):       # 역할에 따라 메시지 출력
            st.markdown(message["content"])

    # 6. 사용자 입력 처리
    if user_input := st.session_state["chat_input"]:
        with st.chat_message("user"):                # 사용자 메시지 출력
            st.markdown(user_input)
        st.session_state.messages.append(           # 세션 상태에 저장
            {"role": "user", "content": user_input}
        )

        # 7. OpenAI 응답 생성 및 스트리밍 출력
        gen = chat_with_bot(st.session_state.messages)  # 전체 대화 전달하여 응답 생성
        with st.chat_message("assistant"):              # 챗봇 응답 출력
            message_placeholder = st.empty()            # 빈 공간 준비
            full_response = ""                          # 응답 내용 누적 변수
            for chunk in gen:                           # 스트리밍 응답 처리
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")  # 실시간 표시
            message_placeholder.markdown(full_response)                # 최종 표시
            st.session_state.messages.append(                         # 응답 저장
                {"role": "assistant", "content": full_response}
            )

# 8. 앱 실행
if __name__ == "__main__":
    main()
