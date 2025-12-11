import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from loader import load_clinic_docs

INDEX_DIR = os.getenv("INDEX_DIR", "index_store")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 60))
TOP_K_DEFAULT = int(os.getenv("TOP_K", 4))

def build_or_load_index(recreate: bool = False):
    index_path = Path(INDEX_DIR)

    embeddings = OpenAIEmbeddings()
    if index_path.exists() and not recreate:
        print("Loading existing FAISS index from", INDEX_DIR)
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)

    print("Building new FAISS index (LangChain)...")
    docs = load_clinic_docs()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(docs_chunks, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print("Index built and saved to", INDEX_DIR)
    return vectorstore

def retrieve_top_k(vectorstore, query: str, k: int = None):
    if k is None:
        k = TOP_K_DEFAULT
    return vectorstore.similarity_search(query, k=k)
