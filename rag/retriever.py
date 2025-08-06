### rag/retriever.py
from vectorstore import similarity_search


def retrieve_docs(query, k=3):
    return similarity_search(query, k)