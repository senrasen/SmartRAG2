
import os
import json
import logging
import time

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Konstanta
CHROMA_PATH = "chromayes"
DATA_DIR = "./data"

# Inisialisasi model
llm = Ollama(model="llama3.2")
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Ekstrak seluruh teks dari metadata JSON
def extract_text_from_json(json_data):
    all_text = []
    for key, value in json_data.items():
        if isinstance(value, str) and value.strip() not in ["-", ""]:
            all_text.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    return "\n".join(all_text)

# Membaca semua file JSON dari folder dan subfolder
def ingest_all_json_in_folder_recursive(folder_path):
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                        text = extract_text_from_json(raw_data)

                        metadata = {
                            "judul": raw_data.get("judul_peraturan", "-"),
                            "nomor": raw_data.get("nomor", "-"),
                            "tahun": raw_data.get("tahun_terbit", "-"),
                            "jenis": raw_data.get("jenis_dokumen", "-"),
                            "singkatan": raw_data.get("singkatan_jenis", "-"),
                            "subjek": raw_data.get("subjek", "-"),
                            "status": raw_data.get("status", "-"),
                            "penandatangan": raw_data.get("penandatangan", "-"),
                        }

                        if text.strip():
                            documents.append(Document(page_content=text, metadata=metadata))
                            logging.info(f"✅ Berhasil proses: {file_path}")
                        else:
                            logging.warning(f"⚠️ Kosong: {file_path}")
                except Exception as e:
                    logging.error(f"❌ Gagal membaca {file_path}: {e}")
    return documents

# Memuat atau membangun vectorstore
def load_vector_db():
    if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
        logging.info("📦 Memuat data dari ChromaDB yang sudah ada...")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    else:
        logging.info("🧱 Membangun ulang vectorstore dari file JSON...")
        data = ingest_all_json_in_folder_recursive(DATA_DIR)
        if not data:
            logging.error("❌ Tidak ada dokumen valid ditemukan.")
            return None
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = splitter.split_documents(data)

        if not documents:
            logging.error("❌ Tidak ada dokumen valid setelah split.")
            return None

        try:
            vectorstore = Chroma.from_documents(documents, embedding=embedding, persist_directory=CHROMA_PATH)
            vectorstore.persist()
            return vectorstore
        except Exception as e:
            logging.error(f"❌ Gagal membuat vectorstore: {e}")
            return None

# Jawab pertanyaan user
def ask_question(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Kamu adalah asisten hukum yang cermat dan hanya menjawab berdasarkan isi dokumen berikut ini.

    Berikut ini adalah informasi dari beberapa dokumen hukum yang relevan:

    {context}

    Petunjuk:
    - Jawablah pertanyaan berdasarkan isi dokumen di atas.
    - Jika memungkinkan, sebutkan sumber dokumen (misal: 'Keputusan Presiden No. 12 Tahun 2021').
    - Jangan menjawab di luar dokumen yang diberikan.
    - Gunakan kalimat formal dan ringkas.

    Pertanyaan: {question}
    Jawaban:
    """
    return llm.invoke(prompt), docs

# --- MAIN ---
vectorstore = load_vector_db()

if vectorstore is None:
    print("Vectorstore tidak tersedia. Pastikan folder './data' berisi file JSON.")
else:
    # Ganti pertanyaan di sini
    question = "Apa isi dari Keputusan Presiden No. 12 Tahun 2021?"

    start = time.time()
    answer, docs = ask_question(vectorstore, question)
    duration = time.time() - start

    print(f"\n==================== Jawaban ====================")
    print(answer)
    print(f"\n🕒 Waktu proses: {duration:.2f} detik")

    print(f"\n📄 Dokumen sumber:")
    for doc in docs:
        meta = doc.metadata
        print(f"- {meta.get('jenis')} {meta.get('nomor')} ({meta.get('tahun')}): {meta.get('judul')}")
