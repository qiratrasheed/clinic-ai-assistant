from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

def build_vector_store(folder_path: str):
    docs = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())

        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        raise ValueError("No documents found in the knowledge base.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"[RAG Tool] {len(chunks)} chunks indexed!")
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 4) -> str:
    results = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context
