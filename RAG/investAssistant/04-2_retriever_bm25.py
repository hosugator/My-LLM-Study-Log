# pip install rank_bm25

# BM25 
# - 정확한 키워드 매칭에 강함
# - 데이터가 전문 용어나 고유명사 위주

from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# 1 환경 변수(.env) 로드 및 OpenAI 키 확인 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2 검색 대상 문서 리스트 준비 
doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",  # 오타 수정: GPD -> GDP
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

# 3 BM25(단어 빈도 기반) 검색기 구성 
#    - 문서별로 source 메타데이터를 다르게 부여(디버깅/추적 편의)
bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": f"bm25_doc_{i}"} for i in range(len(doc_list))]
)
bm25_retriever.k = 1  # 상위 1개 문서만 반환

# 4 임베딩 모델 준비(벡터 검색용) 
#    - 소형: text-embedding-3-small / 고성능: text-embedding-3-large
embedding = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

# 5 문서 임베딩 후 FAISS 벡터스토어 생성 
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": f"faiss_doc_{i}"} for i in range(len(doc_list))]
)

# #6 FAISS를 Retriever로 변환(k 설정) 
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# #7 사용자 질의 정의 
query = "2022년 우리나라 GDP대비 R&D 규모는?"

# #8 두 검색기로 각각 질의 수행 
bm25_docs = bm25_retriever.invoke(query)
faiss_docs = faiss_retriever.invoke(query)

# #9 결과 가독성 있게 출력 
def pretty_print(tag, docs):
    if not docs:
        print(f"{tag} 검색 결과: 검색 결과 없음")
        return
    print(f"{tag} 검색 결과:")
    for i, d in enumerate(docs, 1):
        print(f" [{i}] source={d.metadata.get('source')}")
        print("     ", d.page_content)

pretty_print("BM25", bm25_docs)     # 단어 유사 기반(빠름, 정확 매칭에 강함)
pretty_print("FAISS", faiss_docs)   # 의미 유사 기반(임베딩 품질/도메인 영향)