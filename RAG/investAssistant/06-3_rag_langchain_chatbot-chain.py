# pip install rank_bm25 langchain-openai faiss-cpu python-dotenv

# 1) 라이브러리 임포트 -----------------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import EnsembleRetriever

# 2) API 키 로드 -----------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 3) 문서 데이터 -----------------------------------------------------------------
docs = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다.",
    "삼성전자 2025년 1분기 매출액은 약 7조원으로 잠정 추정됩니다.",
    "2025년 7월 19일 삼성전자 주가는 64,500원입니다."
]
metas = [{"source": f"doc_{i}"} for i in range(len(docs))]

# 4) 간단 retriever 구성(앙상블: BM25 + FAISS) -----------------------------------
bm25 = BM25Retriever.from_texts(docs, metadatas=metas); bm25.k = 1
emb = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
faiss = FAISS.from_texts(docs, emb, metadatas=metas).as_retriever(search_kwargs={"k": 1})
retriever = EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.2, 0.8]) # 단일 검색기 사용하고 싶다면 : EnsembleRetriever(retrievers=bm25)

# 5) 검색 결과를 LLM 컨텍스트 문자열로 변환 --------------------------------------
def format_docs(found):
    return "\n\n".join(d.page_content for d in found) if found else ""

# 6) 프롬프트(자료 없으면 답 못한다고 말하라고 지시) ------------------------------
prompt = PromptTemplate.from_template(
"""아래 '자료'만 근거로 간결히 답하세요.
- 자료 밖 정보를 추측하지 마세요.
- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.
질문:
{question}
자료:
{context}
"""
)
# 7) LLM/파서 준비 ---------------------------------------------------------------
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# 8) 체인 구성(질문 → retriever → 컨텍스트 → 프롬프트 → LLM → 문자열) -------------
# 질문 : prompt 가 완전히 준비되지 않고 검색해서 결과를 받아서 넘겨야 하기 때문에 다음과 같은 방식으로 작성함
chain = (
    {
        "question": RunnablePassthrough(),   # 그냥 받은 입력을 그대로 전달- 원본 질의 내용이 무엇인지 전달 목적
        "context": RunnableLambda(lambda q: format_docs(retriever.invoke(q))),  # 입력을 받아서 가공하는 함수를 적용-retriever로 찾은 자료를 LLM에 같이 전달, 검색된 Document 리스트를 format_docs()로 문자열로 변환, 
    }
    | prompt | llm | parser
)

# 9) 실행 예시 -------------------------------------------------------------------
question = "삼성전자의 올해 매출액은?"
response = chain.invoke(question)
print(response)
