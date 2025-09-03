
# 10_rag_from_sqlite.py
# ---------------------------------------------------------------------------
# 3) (별도 단계로 이미 저장된) SQLite DB 파일에서 문서를 읽어와
#    BM25+FAISS retriever → RAG로 답변 생성
#
# 실행 순서:
#   1) python 10_make_sqlite_data.py   # DB 파일 생성/저장
#   2) python 10_rag_from_sqlite.py    # DB를 로드해 RAG 실행
# ---------------------------------------------------------------------------

import os, sqlite3, json
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever

# 1. .env 로드 및 OpenAI 키 확인 --------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env를 확인하세요.")

client = OpenAI(api_key=api_key)
CHAT_MODEL = "gpt-4o-mini"
EMB_MODEL = "text-embedding-3-small"

# 2. DB 경로 및 로딩 ---------------------------------------------------------
DB_PATH = "company_news.db"
TABLE = "news"

def load_documents(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
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
            "source": f"db_doc_{rid}"
        })
    return texts, metadatas

# 3. 간단 검색기 구성(BM25 + FAISS 앙상블) ----------------------------------
def build_retriever(texts, metadatas):
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas); bm25.k = 2
    emb = OpenAIEmbeddings(api_key=api_key, model=EMB_MODEL)
    faiss_store = FAISS.from_texts(texts, emb, metadatas=metadatas)
    faiss = faiss_store.as_retriever(search_kwargs={"k": 2})
    return EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.3, 0.7])

# 4. 프롬프트 빌더 -----------------------------------------------------------
def build_prompt(query: str, docs):
    if not docs:
        return "제공된 문서에서 찾지 못했습니다."
    ctx = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        ctx.append(
            f"[문서{i}] (source={m.get('source')}, 기업명={m.get('기업명')}, 날짜={m.get('날짜')}, 카테고리={m.get('문서_카테고리')}, 이벤트={m.get('주요_이벤트')})\n"
            f"{d.page_content}"
        )
    context = "\n\n".join(ctx)
    return f"""아래 '자료'만 근거로 한국어로 간결히 답하세요.
- 자료 밖 정보를 추측하지 마세요.
- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.

질문:
{query}

자료:
{context}
"""
# 5. LLM 호출 ---------------------------------------------------------------
def ask_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers in Korean."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# 6. 메인 실행 ---------------------------------------------------------------
if __name__ == "__main__":
    texts, metadatas = load_documents(DB_PATH)
    if not texts:
        print("DB에 문서가 없습니다. 먼저 10_make_sqlite_data.py 를 실행하여 데이터를 채워주세요.")
        raise SystemExit(0)

    retriever = build_retriever(texts, metadatas)

    # 예시 질의
    query = "삼성전자의 2025년 2분기 실적과 주요 이슈는?"
    docs = retriever.invoke(query) or []

    if not docs:
        print("=== 검색 결과 없음 ===")
        print("제공된 문서에서 찾지 못했습니다.")
    else:
        print("=== 검색된 문서 ===")
        for i, d in enumerate(docs, 1):
            print(f"[{i}] source={d.metadata.get('source')} / 기업명={d.metadata.get('기업명')} / 날짜={d.metadata.get('날짜')} / 카테고리={d.metadata.get('문서_카테고리')}")
            print(d.page_content, "\n")

        prompt = build_prompt(query, docs)
        answer = ask_llm(prompt)

        print("=== 최종 답변 ===")
        print(answer)
