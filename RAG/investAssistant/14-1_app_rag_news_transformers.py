# pip install streamlit rank_bm25 faiss-cpu sentence-transformers transformers accelerate

# 1) 라이브러리 임포트 ------------------------------------------------------------
import os, json, sqlite3, torch
import streamlit as st
from langchain_community.retrievers import BM25Retriever                 # 키워드 기반
from langchain_community.vectorstores import FAISS                       # 벡터 저장/검색
from langchain_huggingface import HuggingFaceEmbeddings                  # <- 권장 임베딩
from langchain.retrievers import EnsembleRetriever                       # 앙상블
import transformers                                                      # 로컬 LLM 파이프라인

# 2) DB 경로/테이블 설정 ----------------------------------------------------------
DB_PATH = "company_news.db"
TABLE   = "news"

# 3) DB에서 문서 로드 함수 --------------------------------------------------------
def load_documents_from_sqlite(db_path: str):
    """
    - 반환: texts(list[str]), metadatas(list[dict])
    - texts: 요약 컬럼을 사용하여 임베딩/검색에 활용
    - metadatas: 기업명/날짜/카테고리/이벤트 등 부가정보
    """
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
    # 4-1) BM25: 키워드 기반 검색기 (k=2)
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = 2

    # 4-2) 임베딩(로컬) + FAISS: 의미 기반 검색기 (k=2)
    #      - 한국어/다국어에 무난한 문장 임베딩 모델 예시
    embedding = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    faiss_store = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    faiss = faiss_store.as_retriever(search_kwargs={"k": 2})

    # 4-3) 앙상블: BM25(0.3) + FAISS(0.7) 가중합
    ensemble = EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.3, 0.7])
    return ensemble

# 5) 로컬 LLM 파이프라인 초기화(예: 42dot/42dot_LLM-SFT-1.3B) -------------------
@st.cache_resource(show_spinner=False)
def load_local_pipeline(model_id: str = "42dot/42dot_LLM-SFT-1.3B"):
    """
    - GPU 있으면 float16로, 없으면 float32로 자동 선택
    - 추론(평가) 모드로 설정
    """
    use_cuda   = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    pipe = transformers.pipeline(
        task="text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch_dtype},
        device_map="auto" if use_cuda else None,
    )
    pipe.model.eval()
    return pipe

# 6) 검색 함수(문서 리스트 반환) --------------------------------------------------
def search(query: str, retriever):
    docs = retriever.invoke(query)
    return docs or []

# 7) RAG 프롬프트 구성(문서가 '있을 때만' 호출) ----------------------------------
def build_prompt(query: str, docs):
    """
    - 검색된 문서들을 '자료' 섹션으로 나열
    - ※ 문서 없음 처리 로직은 main에서 담당(LLM 미호출)
    """
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

# 8) LLM 호출(생성) --------------------------------------------------------------
def generate_with_llm(pipe, prompt: str):
    """
    - max_new_tokens / temperature / top_p 등 하이퍼파라미터는 필요에 따라 조정
    - 반환: 모델이 생성한 답변(프롬프트 제외)
    """
    out = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        pad_token_id=pipe.tokenizer.eos_token_id,  # 일부 모델에서 필요
    )
    full = out[0]["generated_text"]
    return full.split("답변:", 1)[-1].strip() if "답변:" in full else full[len(prompt):].strip()

# 9) Streamlit UI 구성 -----------------------------------------------------------
def main():
    st.set_page_config(page_title="🤖 투자 어시스턴트 (RAG)", page_icon="🤖", layout="centered")
    st.title("🤖 투자 어시스턴트 (RAG)")

    # 9-1) DB 로드 및 Retriever / LLM 준비 (캐시)
    try:
        texts, metadatas = load_documents_from_sqlite(DB_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    if "retriever" not in st.session_state:
        st.session_state.retriever = build_ensemble_retriever(texts, metadatas)
    if "pipe" not in st.session_state:
        st.session_state.pipe = load_local_pipeline("42dot/42dot_LLM-SFT-1.3B")

    # 9-2) 채팅 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 9-3) 기존 대화 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 9-4) 입력창
    user_input = st.chat_input("궁금한 점을 물어보세요.")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # (a) 검색
        docs = search(user_input.strip(), st.session_state.retriever)

        # (b) 문서가 없으면 LLM 호출을 '하지 않고' 즉시 고정 답변 반환  ←★ 핵심 변경
        if not docs:
            answer = "제공된 문서에서 찾지 못했습니다."
        else:
            # (c) 문서가 있는 경우에만 프롬프트 생성 후 LLM 호출
            prompt = build_prompt(user_input.strip(), docs)
            answer = generate_with_llm(st.session_state.pipe, prompt)

        # (d) 어시스턴트 응답 출력/저장
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # (선택) 검색된 문서 패널
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

if __name__ == "__main__":
    main()
