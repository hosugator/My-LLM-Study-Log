# pip install streamlit rank_bm25 faiss-cpu sentence-transformers langchain-huggingface
# pip install langchain langchain-community langchain-openai

# 1) 라이브러리 임포트 ------------------------------------------------------------
import os, json, sqlite3
import streamlit as st
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI  # ✅ OpenAI LLM

# 2) DB 경로/테이블 설정 ----------------------------------------------------------
DB_PATH = "company_news.db"
TABLE   = "news"

# 3) DB에서 문서 로드 함수 --------------------------------------------------------
def load_documents_from_sqlite(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"{db_path} 파일이 없습니다. 먼저 데이터 생성 스크립트를 실행하세요.")

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(f"SELECT id, 기업명, 날짜, 문서_카테고리, 요약, 주요_이벤트 FROM {TABLE} ORDER BY 날짜 ASC")
    rows = cur.fetchall()
    conn.close()

    texts, metadatas = [], []
    for rid, company, date, category, summary, events_json in rows:
        texts.append(summary)
        try:
            events = ", ".join(json.loads(events_json))
        except Exception:
            events = events_json
        metadatas.append({
            "id": rid,
            "기업명": company,
            "날짜": date,
            "문서_카테고리": category,
            "주요_이벤트": events,
            "source": f"db_doc_{rid}",
        })
    return texts, metadatas

# 4) 앙상블 Retriever 구성(BM25 + FAISS) ----------------------------------------
def build_ensemble_retriever(texts, metadatas):
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas); bm25.k = 2
    embedding = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    faiss_store = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    faiss = faiss_store.as_retriever(search_kwargs={"k": 2})
    return EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.3, 0.7])

# 5) ✅ OpenAI LLM 초기화(캐시) ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_openai_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    # OPENAI_API_KEY는 환경변수 또는 Streamlit Secrets에 저장
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. Streamlit Secrets 또는 환경변수에 설정하세요.")
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

# 6) 검색 함수(문서 리스트 반환) --------------------------------------------------
def search(query: str, retriever):
    docs = retriever.invoke(query)
    return docs or []

# 7) RAG 프롬프트 구성(문서가 '있을 때만' 호출) ----------------------------------
def build_prompt(query: str, docs):
    lines = []
    lines.append("아래 '자료'만 근거로 한국어로 간결히 답하세요.")
    lines.append("- 자료 밖 정보를 추측하지 마세요.")
    lines.append("- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.\n")
    lines.append(f"질문:\n{query}\n")
    lines.append("자료:")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        lines.append(
            f"[문서{i}] (source={m.get('source')}, 기업명={m.get('기업명')}, 날짜={m.get('날짜')}, "
            f"카테고리={m.get('문서_카테고리')}, 이벤트={m.get('주요_이벤트')})\n{d.page_content}\n"
        )
    lines.append("답변:")
    return "\n".join(lines)

# 8) ✅ OpenAI로 답변 생성 --------------------------------------------------------
def generate_with_llm(llm: ChatOpenAI, prompt: str) -> str:
    resp = llm.invoke(prompt)  # AIMessage
    return resp.content.strip()

# 9) Streamlit UI 구성 -----------------------------------------------------------
def main():
    st.set_page_config(page_title="🤖 투자 어시스턴트 (RAG)", page_icon="🤖", layout="centered")
    st.title("🤖 투자 어시스턴트 (RAG)")

    try:
        texts, metadatas = load_documents_from_sqlite(DB_PATH)
    except FileNotFoundError as e:
        st.error(str(e)); st.stop()

    if "retriever" not in st.session_state:
        st.session_state.retriever = build_ensemble_retriever(texts, metadatas)
    if "llm" not in st.session_state:
        st.session_state.llm = load_openai_llm("gpt-4o-mini", temperature=0.0)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("궁금한 점을 물어보세요.")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        docs = search(user_input.strip(), st.session_state.retriever)

        if not docs:
            answer = "제공된 문서에서 찾지 못했습니다."
        else:
            prompt = build_prompt(user_input.strip(), docs)
            answer = generate_with_llm(st.session_state.llm, prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("🔎 사용한 자료(검색 결과) 보기", expanded=False):
            if not docs:
                st.markdown("_검색 결과 없음_")
            else:
                for i, d in enumerate(docs, 1):
                    m = d.metadata
                    st.markdown(
                        f"**[문서{i}]** (source={m.get('source')}, 기업명={m.get('기업명')}, 날짜={m.get('날짜')}, "
                        f"카테고리={m.get('문서_카테고리')}, 이벤트={m.get('주요_이벤트')})\n\n"
                        f"{d.page_content}"
                    )

# 10) 앱 엔트리포인트 ------------------------------------------------------------
if __name__ == "__main__":
    main()