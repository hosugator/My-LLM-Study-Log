# pip install sentence_transformers : SBERT를 이용하여 문장(문서) 임베딩을 얻을 수 있는 패키지

# https://huggingface.co/models?library=sentence-transformers
# SBERT : 전체 문장을 고정 크기 벡터(embedding)로 변환해주는 Python 라이브러리 - 문장 전체를 하나의 밀집 벡터로 만드는 것이 핵심
# - https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
# - Semantic Similarity (의미적 유사도)
# - Semantic Search (의미 기반 검색)
# - Clustering (군집화)
# - Paraphrase Mining, 분류, 문장 채굴 등 다방면에 응용 가능

# sentence_transformers 라이브러리의 SentenceTransformer, util 가져오기
from sentence_transformers import SentenceTransformer, util

# 1) 사전학습 임베딩 모델 로드 (가볍고 성능 균형: all-MiniLM-L6-v2- 영어 최적화)
# - pretrained model : 다국어 - paraphrase‑multilingual‑mpnet‑base‑v2
# - pretrained model : 한국어 - jhgan/ko‑sbert‑sts, Ko‑SBERT‑Multitask

select_model = "all-MiniLM-L6-v2"

model = SentenceTransformer(select_model)


# 2) 임베딩 생성 함수
def get_embedding_sbert(text: str):
    text = text.strip()

    # numpy array 반환 (convert_to_numpy=True)
    emb = model.encode(
        text, normalize_embeddings=False, convert_to_numpy=True
    )  # 단일 문장이나 여러 문장 둘 다 처리
    return emb


# 3) 코사인 유사도 계산 (SBERT util 사용)
def cosine_similarity(a, b) -> float:  # 3) 코사인 유사도 계산 (SBERT util 사용)
    # util.cos_sim은 torch.Tensor 반환 → float으로 변환
    return float(util.cos_sim(a, b).item())


if __name__ == "__main__":
    s1 = "일찍 학교에 갔다."
    s2 = "날씨가 화창하다."
    s3 = "일찍 초등학교로 등교했다."

    e1 = get_embedding_sbert(s1)  # s1 임베딩
    e2 = get_embedding_sbert(s2)  # s2 임베딩
    e3 = get_embedding_sbert(s3)  # s3 임베딩

    print(e1)  # 임베딩 벡터 (numpy 배열)
    print(
        "[SentenceTransformer] cos(s1, s2) =", cosine_similarity(e1, e2)
    )  # e1, e2 코사인유사도
    print(
        "[SentenceTransformer] cos(s1, s3) =", cosine_similarity(e1, e3)
    )  # e1, e2 코사인유사도
    print(
        "[SentenceTransformer] cos(s2, s3) =", cosine_similarity(e2, e3)
    )  # e1, e2 코사인유사도
