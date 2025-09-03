
# pip install rank_bm25 langchain-openai faiss-cpu

from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
import os

# 1 환경 변수(.env) 로드 및 OpenAI 키 확인 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2 검색 대상 문서 리스트 준비 
doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

# 3 BM25(단어 빈도 기반) 검색기 구성 
bm25_retriever = BM25Retriever.from_texts(doc_list)   # ... code here

bm25_retriever.k = 1  # 상위 1개 문서만

# 4 임베딩 모델 준비(벡터 검색용) 
#    OpenAIEmbeddings- 소형: text-embedding-3-small / 고성능: text-embedding-3-large
embedding = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

# 5 문서 임베딩 후 FAISS 벡터스토어 생성 
faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding=embedding,
    metadatas=[{"source": f"faiss_doc_{i}"} for i in range(len(doc_list))]
)   # ... code here

# 6 FAISS를 Retriever로 변환(k 설정) 
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})   # ... code here

# 7 사용자 질의 정의 
query = "2022년 우리나라 GDP대비 R&D 규모는?"

# 8 앙상블 Retriever 구성(가중치 합 1 권장) 
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

# 9 질의 수행 
ensemble_docs = ensemble_retriever.invoke(query)   # ... code here

# 10 결과 가독성 있게 출력 
def pretty_print(tag, docs):
    if not docs:
        print(f"{tag} 검색 결과: 검색 결과 없음")
        return
    print(f"{tag} 검색 결과:")
    for i, d in enumerate(docs, 1):
        print(f" [{i}] source={d.metadata.get('source')}")
        print("     ", d.page_content)

pretty_print("Ensemble(BM25+FAISS)", ensemble_docs)
