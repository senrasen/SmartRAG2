
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 1. Load Data dari file TXT
def load_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# 2. Potong teks jadi chunk
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    return docs

# 3. Embedding dan simpan ke FAISS
def embed_and_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

# 4. Buat RAG chain
def create_rag_chain(db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Streamlit UI
def main():
    st.title("💬 RAG System with TXT Data")
    uploaded_file = st.file_uploader("Upload TXT file", type="txt")

    if uploaded_file is not None:
        st.success("File uploaded. Processing...")

        # Simpan file sementara
        file_path = f"/tmp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Jalankan pipeline
        raw_text = load_txt_file(file_path)
        docs = split_text(raw_text)
        db = embed_and_store(docs)
        rag_chain = create_rag_chain(db)

        st.success("RAG system ready! Ask your question below:")

        user_question = st.text_input("Enter your question")
        if user_question:
            result = rag_chain.run(user_question)
            st.write("### 📖 Answer")
            st.write(result)

if __name__ == "__main__":
    main()
