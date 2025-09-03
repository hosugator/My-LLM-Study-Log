import os
from dotenv import load_dotenv
from pathlib import Path
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()


def load_document(file_path):
    """지정된 파일 경로에서 문서를 로드합니다."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_document(documents, chunk_size=200, chunk_overlap=50, splitter=None):
    """문서를 청크로 나눕니다. splitter를 사용자 정의할 수 있습니다."""
    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    splits = splitter.split_documents(documents)
    return splits


def create_vector_store(splits, embeddings_model="text-embedding-ada-002"):
    """문서 청크와 임베딩 모델을 사용하여 벡터 스토어를 생성합니다."""
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    return vectorstore


def create_lcel_qa_chain(
    vectorstore,
    llm_model,
    instruction_template,  # 지시 템플릿을 인자로 받도록 수정
    llm_temperature=0,
    llm_max_tokens=1024,
):
    """
    LCEL을 사용하여 RAG 질의응답 체인을 생성합니다.
    """
    retriever = vectorstore.as_retriever()
    
    rag_template = f"""
    다음 문맥을 참고하여 질문에 답하세요.
    질문에 대한 답변은 문맥에 있는 내용만 기반으로 작성하세요.
    {instruction_template}

    {{context}}

    질문: {{question}}
    """
    prompt = ChatPromptTemplate.from_template(rag_template)

    llm = ChatOpenAI(
        model=llm_model,
        temperature=llm_temperature,
        max_completion_tokens=llm_max_tokens,
    )

    parser = StrOutputParser()
    
    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    lcel_qa_chain = (
        {
            "context": itemgetter("query") | retriever | format_docs,
            "question": itemgetter("query"),
        }
        | prompt
        | llm
        | parser
    )

    return lcel_qa_chain


def run_rag_pipeline(
    vectorstore,
    query,
    instruction_template,  # 지시 템플릿을 인자로 받도록 수정
    llm_model="gpt-4o-mini",
    llm_temperature=0.5,
    llm_max_tokens=2048,
):
    """전체 RAG 파이프라인을 실행합니다."""
    qa_chain = create_lcel_qa_chain(
        vectorstore,
        llm_model,
        instruction_template,
        llm_temperature,
        llm_max_tokens,
    )
    
    result = qa_chain.invoke({"query": query})
    return result


if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "The_Adventures_of_Tom_Sawyer.pdf"
    documents = load_document(pdf_path)
    splits = split_document(documents)
    vectorstore = create_vector_store(splits)

    # 올바른 테스트를 위한 프롬프트 지시문 정의
    instruction_templates = {
        "Case 1": '답의 신뢰성이 낮을 때는, 답변에 "신뢰성이 낮습니다"라고 덧붙여.',
        "Case 2": '답의 신뢰성이 낮을 때는, 답변에 신뢰성이 낮습니다라고 덧붙여.',
        "Case 3": '답의 신뢰성이 낮을 때는, 답변에 "신뢰성이 낮습니다"라고 대답해.',
        "Case 4": '답의 신뢰성이 낮을 때는, 답변에 신뢰성이 낮습니다라고 대답해.'
    }

    test_query = "마을 무덤에 있던 남자를 죽인 사람은?"
    repeat_count = 10
    results = {}

    for case_name, instruction_template in instruction_templates.items():
        print(f"\n--- 테스트 시작: {case_name} ---")
        confidence_count = 0
        
        for i in range(repeat_count):
            response = run_rag_pipeline(vectorstore, test_query, instruction_template)
            
            if "신뢰성이 낮습니다" in response:
                confidence_count += 1
            
            print(f"[{i+1}/{repeat_count}] 응답: {response}")

        results[case_name] = confidence_count
        print(f"\n--- 테스트 종료: {case_name} ---")
    
    print("\n\n=== 최종 테스트 결과 요약 ===\n")
    for case_name, count in results.items():
        print(f"[{case_name}] \"신뢰성이 낮습니다\" 포함 횟수: {count}/{repeat_count}")
        
    print("\n------------------------------\n")
    
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(test_query)
    print("\n--- 검색된 문서 내용 (Debug) ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- 문서 #{i+1} ---")
        print(doc.page_content)
    print("-----------------------------\n")