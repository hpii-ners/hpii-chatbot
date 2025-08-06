### rag/utils.py
# Fungsi bantu jika diperlukan nanti (misal: cleaning text, splitting, dsb)

def clean_text(text):
    return text.replace("\n", " ").strip()