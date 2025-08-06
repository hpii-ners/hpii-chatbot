from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_vectorstore():
    if os.path.exists("vectorstore/index.faiss"):
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local("vectorstore", embedding)
    return None

def rebuild_vectorstore(pdf_path="docs/your_reference.pdf"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embedding)
    db.save_local("vectorstore")
    return db
