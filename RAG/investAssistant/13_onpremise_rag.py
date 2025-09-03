# pip install rank_bm25 faiss-cpu sentence-transformers transformers accelerate
# pip install langchain-openai langchain-community python-dotenv

# 1. 라이브러리 임포트 -----------------------------------------------------------
import os, json, sqlite3, torch
from dotenv import load_dotenv

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import EnsembleRetriever

import transformers

# 2. 환경 로드 (필수는 아님: on-premise이므로 API키 필요없음) --------------------
load_dotenv()

# 3. DB 경로/테이블 --------------------------------------------------------------
DB_PATH = "company_news.db"
TABLE   = "news"

# 4. DB에서 문서 로드 ------------------------------------------------------------
def load_documents(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"{db_path} 파일이 없습니다. 먼저 '10_make_sqlite_data.py'를 실행해 데이터를 생성하세요."
        )

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
            "id": rid, "기업명": company, "날짜": date,
            "문서_카테고리": category, "주요_이벤트": events,
            "source": f"db_doc_{rid}"
        })
    return texts, metadatas

# 5. on-premise 임베딩(SBERT) & FAISS/BM25/앙상블 구성 ---------------------------
def build_retriever(texts, metadatas):
    # 5-1) BM25 (키워드 기반)
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = 2

    # 5-2) SBERT 임베딩 (로컬)  ※ 다국어: distiluse-base-multilingual-cased-v1
    embedding = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v1")

    # 5-3) FAISS 벡터스토어 (의미기반)
    faiss_store = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    faiss = faiss_store.as_retriever(search_kwargs={"k": 2})

    # 5-4) 앙상블 (가중치 예: 키워드 0.3, 의미 0.7)
    ensemble = EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.3, 0.7])
    return ensemble

# 6. on-premise LLM 파이프라인(Hugging Face Transformers) -----------------------
def build_pipeline(model_id: str = "42dot/42dot_LLM-SFT-1.3B"):
    # GPU 있으면 float16, 없으면 float32
    use_cuda   = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    pipe = transformers.pipeline(
        task="text-generation",
        model=model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if use_cuda else None,  # GPU면 자동 배치
    )
    pipe.model.eval()
    return pipe

# 7. 검색 함수 -------------------------------------------------------------------
def search(query: str, retriever):
    return retriever.invoke(query) or []

# 8. 프롬프트 구성 ---------------------------------------------------------------
def build_prompt(query: str, docs):
    if not docs:
        return "제공된 문서에서 찾지 못했습니다."
    lines = [
        "아래 '자료'만 근거로 한국어로 간결히 답하세요.",
        "- 자료 밖 정보를 추측하지 마세요.",
        "- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.",
        "",
        f"질문:\n{query}\n",
        "자료:"
    ]
    for i, d in enumerate(docs, 1):
        m = d.metadata
        lines.append(
            f"[문서{i}] (source={m.get('source')}, 기업명={m.get('기업명')}, 날짜={m.get('날짜')}, "
            f"카테고리={m.get('문서_카테고리')}, 이벤트={m.get('주요_이벤트')})\n{d.page_content}\n"
        )
    lines.append("답변:")
    return "\n".join(lines)

# 9. LLM 호출 --------------------------------------------------------------------
def sllm_generate(pipe, prompt: str):
    gen = pipe(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        pad_token_id=pipe.tokenizer.eos_token_id,  # 일부 모델에서 필요
    )
    # 일부 모델은 전체 프롬프트+생성이 합쳐져 반환 → 프롬프트 이후만 추출
    full_text = gen[0]["generated_text"]
    # 간단 split: 마지막 "답변:" 이후 텍스트를 답변으로 사용
    if "답변:" in full_text:
        return full_text.split("답변:", 1)[-1].strip()
    return full_text.strip()

# 10. 메인 실행 ------------------------------------------------------------------
if __name__ == "__main__":
    # (1) DB에서 로드
    texts, metadatas = load_documents(DB_PATH)
    if not texts:
        print("DB에 문서가 없습니다. 먼저 데이터 생성 스크립트를 실행해주세요.")
        raise SystemExit(0)

    # (2) retriever 구성(BM25+FAISS on-premise 임베딩)
    retriever = build_retriever(texts, metadatas)

    # (3) on-premise LLM 준비
    pipeline = build_pipeline(model_id="42dot/42dot_LLM-SFT-1.3B")

    # (4) 질의 → 검색 → 프롬프트 → LLM
    query = "삼성전자의 2025년 2분기 실적과 주요 이슈는?"
    docs = search(query, retriever)

    print("=== 검색된 문서 ===")
    if not docs:
        print("(검색 결과 없음)")
        print("제공된 문서에서 찾지 못했습니다.")
    else:
        for i, d in enumerate(docs, 1):
            print(f"[{i}] source={d.metadata.get('source')} / 기업명={d.metadata.get('기업명')} / 날짜={d.metadata.get('날짜')} / 카테고리={d.metadata.get('문서_카테고리')}")
            print(d.page_content, "\n")

        prompt = build_prompt(query, docs)
        answer = sllm_generate(pipeline, prompt)

        print("=== 최종 답변 ===")
        print(answer)
