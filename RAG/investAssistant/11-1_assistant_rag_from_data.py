# pip install rank_bm25 langchain-openai faiss-cpu python-dotenv pykrx

from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
from openai import OpenAI
from pykrx import stock
from datetime import datetime, timezone, timedelta
import json, os

# #1 .env 로드 및 OpenAI 클라이언트 준비 ---------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_OPENAI_API_KEY") or os.getenv(
    "OPENAI_API_KEY"
)  # 키 변수명 혼용 대비

client = OpenAI(api_key=api_key)
model = "gpt-4o-mini"  # 사용할 OpenAI 모델

# 2 (예시) DB에서 가져왔다고 가정한 뉴스 데이터 -------------------------------
data = [
    {
        "기업명": "삼성전자",
        "날짜": "2024-03-02",
        "문서 카테고리": "인수합병",
        "요약": "삼성전자가 HVAC(냉난방공조) 사업 인수를 타진 중이며, 이는 기존 가전 사업의 약점 보완을 목적으로 한다.",
        "주요 이벤트": ["기업 인수합병"],
    },
    {
        "기업명": "삼성전자",
        "날짜": "2024-03-24",
        "문서 카테고리": "인수합병",
        "요약": "테스트 하나 둘 셋",
        "주요 이벤트": ["신제품 출시"],
    },
    {
        "기업명": "현대차",
        "날짜": "2024-04-02",
        "문서 카테고리": "인수합병",
        "요약": "삼성전자가 HVAC(냉난방공조) 사업 인수를 타진 중이며, 이는 기존 가전 사업의 약점 보완을 목적으로 한다.",
        "주요 이벤트": ["기업 인수합병", "신제품 출시"],
    },
]
doc_list = [item["요약"] for item in data]

# 3 키워드(BM25) 검색기 준비 ---------------------------------------------------
bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": i} for i in range(len(data))]
)
bm25_retriever.k = 1  # 상위 1개만

# 4 임베딩/FAISS(벡터) 검색기 준비 --------------------------------------------
embedding = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": i} for i in range(len(data))]
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# 5 앙상블 검색기 구성(BM25:FAISS = 0.2:0.8) ---------------------------------
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.2, 0.8]
)


# 6 공통 LLM 호출 함수 ---------------------------------------------------------
def chatgpt_generate(query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return resp.choices[0].message.content.strip()


# 7 (개선) 기업명 → 티커 추출(LLM JSON 강제) ----------------------------------
def first_chain(query: str) -> dict:
    """
    - 기업명이 포함되어 있으면 {"기업명":"티커"} JSON만 출력하도록 강제
    - 없으면 빈 객체 {} 출력
    - 주의: LLM 신뢰성 한계 → 실제 운영은 pykrx 매핑 사용 권장
    """
    prompt = (
        "아래 질문에 특정 한국 상장사 기업명이 포함되어 있으면, "
        "그 기업명과 6자리 티커를 JSON 객체로만 출력하시오. "
        "기업명이 없으면 빈 객체 {}만 출력하시오.\n"
        '출력 예시: {"삼성전자":"005930"}\n\n'
        f"질문: {query}"
    )
    txt = chatgpt_generate(prompt)
    try:
        return json.loads(txt)  # ← eval 대신 안전한 JSON 파싱
    except json.JSONDecodeError:
        return {}


# 8 오늘(KST) 날짜의 종가 가져오기(pykrx) ------------------------------------
def get_today_close_price(ticker: str) -> str:
    """오늘이 휴장일/데이터 없음이면 '모름' 반환"""
    today_kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv(today_kst, today_kst, ticker)
        if df is not None and len(df) > 0:
            return str(df.iloc[0]["종가"])  # pykrx 컬럼명: 시가/고가/저가/종가/...
        return "모름"
    except Exception:
        return "모름"


# 9 뉴스 검색 + 가격 포함 프롬프트 생성 ---------------------------------------
def search(query: str):
    return ensemble_retriever.invoke(query) or []


def prompt_and_generate(query: str, docs, price: str):
    prompt = (
        "아래 질문을 기반으로 검색된 뉴스를 참고하여 질문에 대한 답을 한국어로 간결히 작성하시오. "
        "답변 마지막에 오늘의 종가 정보를 포함하시오.\n\n"
        f"질문: {query}\n"
        f"오늘의 종가: {price}\n\n"
    )
    for i, d in enumerate(docs, 1):
        idx = d.metadata["source"]
        prompt += (
            f"[뉴스{i}]\n"
            f"요약: {d.page_content}\n"
            f"카테고리: {data[idx]['문서 카테고리']}\n"
            f"이벤트: {', '.join(data[idx]['주요 이벤트'])}\n\n"
        )
    return chatgpt_generate(prompt)


# 10 실행 ----------------------------------------------------------------------
query = "삼성전자가 인수하는 기업은?"
company_map = first_chain(query)  # {"삼성전자":"005930"} 형태 기대(없으면 {})
tickers = list(company_map.values())
price = get_today_close_price(tickers[0]) if tickers else "모름"

retrieved = search(query)  # 앙상블 검색
answer = prompt_and_generate(query, retrieved, price)
print(answer)
