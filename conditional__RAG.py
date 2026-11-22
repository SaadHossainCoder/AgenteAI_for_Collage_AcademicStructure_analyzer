import os 
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Stap- 01 Embedding model
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Stap- 02 Build RAG Retriver 
def build_retriver(pdf_path : str) :
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)

    chunk = splitter.split_documents(document)

    vectorstor = FAISS.from_documents(chunk, embeddings)

    return vectorstor.as_retriever(search_kwargs = {"k": 4})


accedemic_retriver = build_retriver("./asset/academics_handbook.pdf")
fee_retriver = build_retriver("./asset/fee_structure.pdf")