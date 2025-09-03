# pip install scikit-learn

# 1. 라이브러리 임포트 
from langchain_community.retrievers import BM25Retriever       # BM25 검색기
from langchain_openai import OpenAIEmbeddings                  # OpenAI 임베딩
from langchain_community.vectorstores import FAISS             # FAISS 벡터 스토어
from langchain.retrievers import EnsembleRetriever             # 앙상블 검색기
from dotenv import load_dotenv
import os
from sklearn.metrics import precision_score, recall_score, f1_score  # 평가 지표

# 2. 환경 변수 로드 및 API 키 설정 
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# 3. 검색 대상 문서 데이터 준비 
doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 국내총생산 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

# 4. 평가용 정답 데이터(gold standard) 정의 
#    - key: 질의(query), value: 해당 정답 문서의 인덱스 리스트
gold_data = {
    "코로나가 유행한 연도": [0],
    "2022년 GDP 대비 R&D 예산": [2],
    "2024년의 국내총생산 전망": [1]
}

# 5. BM25 검색기 구성 
bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": i} for i in range(len(doc_list))]
)
bm25_retriever.k = 1  # 가장 유사한 문서 1개만 반환

# 6. 벡터 기반 검색기(FAISS) 구성 
embedding = OpenAIEmbeddings(api_key=api_key)

faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": i} for i in range(len(doc_list))]
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# 7. 앙상블 검색기 구성 
#    - retrievers: BM25와 FAISS를 함께 사용
#    - weights: 각 검색기의 가중치
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.2, 0.8]
)

# 8. 질의어에 대한 검색 수행 
retriever_docs = {
    query: ensemble_retriever.invoke(query) for query in gold_data
}
print(f"retriever 검색 결과: {retriever_docs if retriever_docs else '검색 결과 없음'}")

# 9. 검색기 성능 평가 함수 정의 
def evaluate_retriever(retriever_docs, gold_standard, documents):
    precisions = []  # 정밀도 리스트
    recalls = []     # 재현율 리스트
    f1_scores = []   # F1 점수 리스트
    
    # 각 질의별로 평가 수행
    for query in gold_standard:
        retrived = [doc.metadata['source'] for doc in retriever_docs[query]]
        gold = gold_standard[query]
        
        # y_true: 정답 문서 인덱스 위치는 1, 나머지는 0
        y_true = [1 if i in gold else 0 for i in range(len(documents))]
        # y_pred: 검색된 문서 인덱스 위치는 1, 나머지는 0
        y_pred = [1 if i in retrived else 0 for i in range(len(documents))]
    
        # 정밀도, 재현율, F1 점수 계산
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 결과 저장
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
    # 평균 점수 계산
    avg_precisions = sum(precisions) / len(gold_standard)
    avg_recalls = sum(recalls) / len(gold_standard)
    avg_f1_scores = sum(f1_scores) / len(gold_standard)

    return avg_precisions, avg_recalls, avg_f1_scores

# 10. 평가 함수 실행 및 결과 출력 
avg_precisions, avg_recalls, avg_f1_scores = evaluate_retriever(
    retriever_docs, gold_data, doc_list
)
print(f"평균 정밀도: {avg_precisions:.4f}, 평균 재현율: {avg_recalls:.4f}, 평균 F1 점수: {avg_f1_scores:.4f}")
