import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ sudah diperbaiki
from langchain_community.vectorstores import FAISS

# =========================================================
# Fungsi build_index()  (sudah ada)
# =========================================================
def build_index(data_folder="pdf_files"):
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    for filename in os.listdir(data_folder):
        path = os.path.join(data_folder, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif filename.endswith(".csv"):
            loader = CSVLoader(path)
        else:
            print(f"⚠️ Format tidak dikenali, dilewati: {filename}")
            continue

        print(f"📄 Memuat: {path}")
        try:
            docs = loader.load()
        except Exception as e:
            print(f"❌ Gagal memuat {filename}: {e}")
            continue

        chunks = splitter.split_documents(docs)
        print(f"➡️  {filename} menghasilkan {len(chunks)} chunk")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("❌ Tidak ada file yang valid ditemukan atau teks tidak terbaca.")

    # Gunakan model embedding multilingual
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # Simpan FAISS index
    db = FAISS.from_documents(all_chunks, embedding)
    db.save_local("faiss_index")
    print(f"✅ Index FAISS berhasil dibuat dari total {len(all_chunks)} chunk dokumen.")

# =========================================================
# Tambahan baru: load_vectorstore()
# =========================================================
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    raise FileNotFoundError("Index belum ada, jalankan build_index() dulu.")

# =========================================================
# Tambahan baru: similarity_search()
# =========================================================
def similarity_search(query: str, k: int = 3):
    """
    Menerima query string dan mengembalikan k potongan teks terdekat.
    """
    db = load_vectorstore()
    docs = db.similarity_search(query, k=k)
    return [d.page_content for d in docs]

# =========================================================
# Tambahan baru: store_documents()
# =========================================================
def store_documents(documents):
    """
    Menerima list dict {'text': ...} dan menyimpannya ke dalam index.
    """
    texts = [d["text"] for d in documents if d.get("text", "").strip()]
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    db = FAISS.from_texts(texts, embeddings)
    db.save_local("faiss_index")
    print(f"✅ {len(texts)} dokumen baru berhasil disimpan ke index.")
