from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def load_document(file_path):
    """지정된 파일 경로에서 문서를 로드합니다."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_document(documents, chunk_size, chunk_overlap):
    """문서를 청크로 나눕니다."""
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return r_splitter.split_documents(documents=documents)


def create_vector_store(splits, embeddings_model):
    """문서 청크와 임베딩 모델을 사용하여 벡터 스토어를 생성합니다."""
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    return FAISS.from_documents(documents=splits, embedding=embeddings)


def create_qa_chain(vectorstore, model, temperature, max_tokens):
    """검색 기반 질의응답 체인을 생성합니다."""
    retriever = vectorstore.as_retriever()
    chat = ChatOpenAI(
        temperature=temperature,
        model=model,
        max_completion_tokens=max_tokens,
    )
    qa_chain = RetrievalQA.from_chain_type(
        chain_type="stuff",
        llm=chat,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def run_rag_pipeline(
    file_path,
    chunk_size,
    chunk_overlap,
    embeddings_model,
    llm_model,
    llm_temperature,
    llm_max_tokens,
    query,
):
    """전체 RAG 파이프라인을 실행합니다."""
    documents = load_document(file_path)
    splits = split_document(documents, chunk_size, chunk_overlap)
    vectorstore = create_vector_store(splits, embeddings_model)
    qa_chain = create_qa_chain(
        vectorstore, llm_model, llm_temperature, llm_max_tokens
    )
    result = qa_chain.invoke({"query": query})

    print(f"query: {query}")
    print(f"result: {result['result']}")
    print(f'source documents num: {len(result["source_documents"])}')
    return result

if __name__ == "__main__":
    raw_data = "The_Adventures_of_Tom_Sawyer.pdf"
    n_chunk_size = 200
    n_chunk_overlap = 50
    select_embeddings_model = "text-embedding-ada-002"
    select_model = "gpt-4o-mini"
    max_tokens = 2048
    select_temperature = 0.5
    query = "마을 무덤에 있던 남자를 죽인 사람은?"

    run_rag_pipeline(
        raw_data,
        n_chunk_size,
        n_chunk_overlap,
        select_embeddings_model,
        select_model,
        select_temperature,
        max_tokens,
        query,
    )
