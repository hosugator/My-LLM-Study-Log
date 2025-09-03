# 1. 라이브러리 및 API 키 설정
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# RAG 파이프라인 모듈 임포트
from module_RAG import run_rag_pipeline

# 2. GPT 모델로 단일 응답을 생성하는 함수 정의
# RAG 파이프라인으로 대체하므로 chat_single_turn 함수는 더 이상 필요하지 않습니다.


# 3. Streamlit 앱 UI 구성
def main():
    st.title("RAG Chatbot with Streamlit and OpenAI")

    # 사용자에게 PDF 파일 업로드를 요구
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

    # 4. 사용자 입력 받기
    prompt = st.text_input("질문을 입력하세요:")

    # 5. 버튼 클릭 시 RAG 파이프라인 실행
    if st.button("질문하기") and prompt:
        with st.spinner("RAG 파이프라인이 답변을 생성 중입니다..."):

            if uploaded_file is not None:
                # 업로드된 파일을 임시 파일로 저장하여 경로를 run_rag_pipeline에 전달
                # 이 과정이 필요합니다.
                temp_dir = "temp_dir"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # run_rag_pipeline 함수 호출 (RAG 파이프라인 실행)
                try:
                    response = run_rag_pipeline(temp_file_path, prompt)

                    st.markdown("### GPT의 답변:")
                    st.write(response["result"])  # LCEL 파이프라인이 반환한 결과 출력

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
            else:
                st.warning("먼저 PDF 파일을 업로드해야 합니다.")


# 앱 실행
if __name__ == "__main__":
    main()
