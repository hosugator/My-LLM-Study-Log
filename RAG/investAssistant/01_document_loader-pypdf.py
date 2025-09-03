# pip install langchain langchain-community pypdf

#### pdf 파일에 있는 텍스트 추출하기 : pdf 파싱하여 페이지단위로 가져옴
# 파일준비 https://www.bok.or.kr/imer/singl/rsrchrData/list.do?pageIndex=1&targetDepth=&menuNo=500216&syncMenuChekKey=2&searchCnd=1&searchKwd=&depth2=500834&date=&sdate=&edate=&sort=1&pageUnit=10 혁신과 경제성장=> bok_sample.pdf
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


pdf_path = Path(__file__).parent / "bok_sample.pdf"
loader = PyPDFLoader(pdf_path)  # ... code here

#pages = loader.load()  # ... code here
pages = loader.load_and_split()

print(len(pages))
text = pages[0].page_content
print(text)