# pip install rank_bm25
# ensemble_search.py 코드 가져오기

from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever 
from dotenv import load_dotenv
import os
from openai import OpenAI

# 1 .env 로드 및 OpenAI 키 확인 ------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2 데이터 준비 ------------------------------------------------------
doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다.",
    "삼성전자 2025년 1분기 매출액은 약 7조원으로 잠정 추정됩니다.",
    "2025년 7월 19일 삼성전자 주가는 64,500원입니다."
]
metadatas = [{"source": f"doc_{i}"} for i in range(len(doc_list))]  # 문서별 구분

# 3 BM25(단어 기반) 검색기 구성 -----------------------------------------------
bm25_retriever = BM25Retriever.from_texts(  # ... code here
    doc_list,
    metadatas=metadatas
)
bm25_retriever.k = 1  # 상위 1개만

# 4 임베딩 모델 준비(벡터 검색용) --------------------------------------------
#   - 가벼운 기본값: text-embedding-3-small (필요 시 large로 변경)
embedding = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small"
)

# 5 FAISS 벡터스토어 + 벡터 기반 검색기 ---------------------------------------
faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding,
    metadatas=metadatas
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# 6 앙상블 검색기 구성(가중치 합이 1에 가까우면 해석 용이) --------------------
#   - 의미 유사(임베딩) 비중을 더 높게: 0.2(BM25) : 0.8(FAISS)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.2, 0.8]
)

# 7 검색 함수: 항상 '문서 리스트'를 반환(검색 결과 없으면 빈 리스트) -----------
def search(query: str):
    docs = ensemble_retriever.invoke(query)
    return docs or []

# 8 OpenAI 클라이언트/모델 준비 ----------------------------------------------
client = OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"

# 9 프롬프트 구성 & 답변 생성 ---------------------------------------------------
def build_prompt(query: str, docs_texts):
    # 문서들 사이에 개행을 넣어 가독성/정확도 향상
    context = "\n\n".join([f"[문서{i+1}]\n{t}" for i, t in enumerate(docs_texts)])
    prompt = f"""아래 '자료'만 근거로 한국어로 간결히 답하세요.
- 자료 밖 정보를 추측하지 마세요.
- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.

질문:
{query}

자료:
{context}
"""
    return prompt

def ask_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers in Korean."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# #10 실행: 검색 → 문서 컨텍스트로 답변 생성 -----------------------------------
query = "삼성전자의 올해 매출액은?"
retrieved_docs = search(query)                          # Document 객체 리스트
doc_texts = [d.page_content for d in retrieved_docs]    # LLM에 넣을 텍스트만 추출
prompt = build_prompt(query, doc_texts)
answer = ask_llm(prompt)

# 결과 확인
print("=== 검색된 문서 ===")
for i, d in enumerate(retrieved_docs, 1):
    print(f"[{i}] source={d.metadata.get('source')}")
    print(d.page_content, "\n")

print("=== 최종 답변 ===")
print(answer)