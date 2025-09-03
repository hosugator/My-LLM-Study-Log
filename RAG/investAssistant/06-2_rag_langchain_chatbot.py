# pip install langchain-openai faiss-cpu  (GPU 사용 여부에 따라 선택 - faiss-gpu는 python 3.8 버전에서 지원)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# 1 .env 로드 및 API 키 확보 -------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2 임베딩 모델 준비 ---------------------------------------------------------
#   - 비용/속도 균형: text-embedding-3-small
#   - 더 높은 성능 필요 시: text-embedding-3-large
embedding_model = OpenAIEmbeddings(
    api_key=api_key,
    model='text-embedding-3-small'                                
                                   )  # ... code here

# 3 PDF (bok_sample.pdf) 로드 (페이지 단위 Document 리스트) -----------------------------------
loader =  PyPDFLoader('bok_sample.pdf')  # ... code here
documents = loader.load()  # ... code here   # 전체 문서(페이지별 Document 목록)

# 4 텍스트 청크 분할기 설정 --------------------------------------------------
#   - separator: 문장 경계(예: '. ', '\n')에 맞게 조정
#   - chunk_size = 500 / chunk_overlap=100 : 검색 품질과 길이 제한을 고려해 조정
#   - length_function = len
text_splitter = CharacterTextSplitter(
    separator='.',
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len       
    )  # ... code here

# 5 페이지 문서(documents)를 청크로 변환 ------------------------------------------------
texts = text_splitter.split_doucments(documents)  # ... code here
# print(texts[0]); print(len(texts))  # 필요 시 확인

# 6 벡터스토어(FAISS) 구축 ---------------------------------------------------
#   - 각 청크를 임베딩하여 인덱스 생성
db = FAISS.from_documents(texts, embedding_model)  # ... code here # 예: 167개 청크 임베딩

# 7 Retriever로 변환 (검색기) ------------------------------------------------
#   - search_type: "similarity"(기본) / "mmr" 등
#   - k:3 상위 몇 개 결과를 가져올지
retriever =  db.as_retriever(search_type='similarity', search_kwards={"k":3}) # ... code here

# 8 질의(Query) 실행 ---------------------------------------------------------
query = "2022년 우리나라 GDP대비 R&D 규모는?"
docs = retriever.invoke(query)  # ... code here   # 유사한 청크 상위 k개 반환

# #9 결과 출력(가독성 있게) ---------------------------------------------------
for i, d in enumerate(docs, 1):
    print(f"\n[{i}] score: 제공 안됨(FAISS 기본), source: {d.metadata.get('source')}, page: {d.metadata.get('page')}")
    print(d.page_content[:400].strip(), "...")