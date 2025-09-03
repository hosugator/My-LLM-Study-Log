## 청킹과정 : 입력되는 텍스트의 길이를 조절하기위해 큰 문서를 작은 단위로 쪼개기 = 글자수, 의미 단위, 문단 단위 기준으로
# 03_langchain_splitter.py
# langchain-text-splitters : langchain 과 langchain-community 설치하면 사용할 수 있음

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


# #1 PDF 로더 준비: 파일 경로를 지정해 로더 생성
loader = PyPDFLoader("bok_sample.pdf")

# #2 PDF 로드(페이지 단위 Document 리스트 반환)
pages = loader.load()   # ... code here # 필요 시 loader.load_and_split()도 가능

# #3 예시: 첫 번째 페이지의 텍스트만 추출
text = pages[0].page_content
print(len(text))  # 첫 페이지 텍스트 길이(문자 수) 확인

# #4 문자 기반 청크 분할기 설정
#    - separator: 문장 구분자(예: '.', '. ', '\n' 등)
#    - chunk_size: 청크 최대 길이(문자 기준)
#    - chunk_overlap: 청크 간 겹치는 길이(문자 기준)
#    - length_function: 길이 계산 함수(기본은 len; 토큰 기준으로 바꿀 수도 있음)

text_splitter = CharacterTextSplitter(   # ... code here
    separator='.',        # 필요에 따라 '. ' 또는 '\n' 등으로 조정
    chunk_size=500,
    chunk_overlap=100,
    length_function=len    # 토큰 기준으로 바꾸려면 tiktoken 등으로 함수 교체
)  # RecursiveCharacterTextSplitter는 기본값이 chunk_size=1000, chunk_overlap=200 ( 하지만 500 +/- 100 정도 적당)

# #5 텍스트를 청크로 분할(첫 페이지 텍스트 대상)
texts = text_splitter.split_text(text)  # ... code here

# #6 분할 결과 확인
print(texts[0])        # 첫 번째 청크 내용
print(len(texts[0]))   # 첫 번째 청크 길이(문자 수)
print(len(texts))      # 생성된 청크 개수