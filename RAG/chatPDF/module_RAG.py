from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

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


def create_qa_chain(
    vectorstore, model, temperature=0.5, max_tokens=1024, chain_type="stuff"
):
    """검색 기반 질의응답 체인을 생성합니다."""
    retriever = vectorstore.as_retriever()
    chat = ChatOpenAI(
        temperature=temperature,
        model=model,
        max_completion_tokens=max_tokens,
    )
    qa_chain = RetrievalQA.from_chain_type(
        chain_type=chain_type,
        llm=chat,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def run_rag_pipeline(
    file_path,
    query,
    chunk_size=200,
    chunk_overlap=50,
    embeddings_model="text-embedding-ada-002",
    splitter=None,
    llm_model="gpt-4o-mini",
    llm_temperature=0.5,
    llm_max_tokens=2048,
    qa_chain_type="stuff",
):
    """전체 RAG 파이프라인을 실행합니다."""

    documents = load_document(file_path)
    splits = split_document(documents, chunk_size, chunk_overlap, splitter)
    vectorstore = create_vector_store(splits, embeddings_model)
    qa_chain = create_qa_chain(
        vectorstore, llm_model, llm_temperature, llm_max_tokens, qa_chain_type
    )
    result = qa_chain.invoke({"query": query})

    print(f"query: {query}")
    print(f"result: {result['result']}")
    print(f'source documents num: {len(result["source_documents"])}')
    return result


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "The_Adventures_of_Tom_Sawyer.pdf")
    raw_data = pdf_path
    query = "마을 무덤에 있던 남자를 죽인 사람은?"

    # 기본값을 사용하여 필수 매개변수만 전달
    run_rag_pipeline(raw_data, query)

    # 기본값 중 일부를 재정의하여 호출
    run_rag_pipeline(raw_data, query, chunk_size=300, llm_model="gpt-4")
