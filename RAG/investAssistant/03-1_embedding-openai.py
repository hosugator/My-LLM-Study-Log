# OpenAIEmbeddings + 직접 코사인 유사도 (LangChain)

# ==== ① OpenAIEmbeddings + numpy 코사인 ====
# pip install python-dotenv langchain-openai numpy

import os
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SimilarityFunction

# .env에서 OPENAI_API_KEY 로드
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")     # OPENAI는 load_dotenv()만으로 자동 인식

# 1) 임베딩 백엔드 (OpenAI Embeddings)
emb = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

# 2) 임베딩 생성 함수
def get_embedding_openai(text: str) -> np.ndarray:
    v = emb.embed_query(text.strip())  # list[float] # .embed_query(text) → 단일 문장 임베딩, .embed_documents(list_of_texts) → 여러 문장 임베딩
    return np.array(v, dtype=np.float32)

# 3-1) 코사인 유사도 계산 (SBERT util 사용)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """직접 구현한 코사인 유사도 (안정성 위해 0벡터 예외 처리)"""
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# 3-2) 코사인 유사도 계산 (SBERT util 사용)
similarity_fn = SimilarityFunction.to_similarity_fn("cosine") # SimilarityFunction내의 유사도 함수 정의

if __name__ == "__main__":
    s1 = "일찍 학교에 갔다."
    s2 = "날씨가 화창하다."
    s3 = "일찍 초등학교로 등교했다."

    e1 = get_embedding_openai(s1)
    e2 = get_embedding_openai(s2)
    e3 = get_embedding_openai(s3)

    print("[OpenAIEmbeddings] cos(s1, s2) =", cosine_similarity(e1, e2))
    print("[OpenAIEmbeddings] cos(s1, s3) =", cosine_similarity(e1, e3))
    print("[OpenAIEmbeddings] cos(s2, s3) =", cosine_similarity(e2, e3))


    #print(e1)  # 임베딩 벡터 (numpy 배열)
    score_1_2 = similarity_fn(e1, e2)
    score_1_3 = similarity_fn(e1, e3)
    score_2_3 = similarity_fn(e2, e3)

    print("[SentenceTransformer] cos(s1, s2) =", score_1_2)
    print("[SentenceTransformer] cos(s1, s3) =", score_1_3)
    print("[SentenceTransformer] cos(s2, s3) =", score_2_3)
    
