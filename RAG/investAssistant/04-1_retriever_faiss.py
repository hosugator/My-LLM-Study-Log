# pip install langchain-openai faiss-cpu

# FAISS
# - 데이터가 서술형·의미 변형이 많음

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
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

# 3 임베딩 모델 준비(벡터 검색용) 
#    - 소형: text-embedding-3-small / 고성능: text-embedding-3-large
embedding = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

# 4 문서 임베딩 후 FAISS 벡터스토어 생성 
# from_texts() or from_documents() : page_content + metadata 
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": f"faiss_doc_{i}"} for i in range(len(doc_list))]
)

# 5 FAISS를 Retriever로 변환(k 설정) 
retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# 간단하고 직접적인 검색 : 테스트할 때
#results = faiss_vectorstore.similarity_search_with_score("2022년 우리나라 GDP대비 R&D 규모는?", k=2)
results = faiss_vectorstore.similarity_search_with_score("2022년 우리나라의 GDP 대비 연구개발(R&D) 지출 비율은 몇 퍼센트인가요?", k=2)

for i, (doc, distance) in enumerate(results, 1):
    similarity_est = 1 - distance  # 정규화된 임베딩일 때만 유효
    print(f"--- Result {i} ---")
    print(f"Cosine Distance: {distance:.4f}")
    print(f"Estimated Similarity: {similarity_est:.4f}")
    print(f"Text: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()


# 6 사용자 질의 정의 
query = "2022년 우리나라 GDP대비 R&D 규모는?"
#query = "2022년 우리나라의 GDP 대비 연구개발(R&D) 지출 비율은 몇 퍼센트인가요?"

# 7 질의 수행 
docs = retriever.invoke(query)

# 8 결과 가독성 있게 출력 
def pretty_print(tag, docs):
    if not docs:
        print(f"{tag} 검색 결과: 검색 결과 없음")
        return
    print(f"{tag} 검색 결과:")
    for i, d in enumerate(docs, 1):
        print(f" [{i}] source={d.metadata.get('source')}")
        print("     ", d.page_content)

pretty_print("FAISS", docs)
