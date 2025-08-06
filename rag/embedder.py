### rag/embedder.py
from vectorstore import store_documents


def embed_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pisah per paragraf
    chunks = content.split("\n\n")
    documents = [{"text": chunk} for chunk in chunks if chunk.strip()]
    store_documents(documents)
