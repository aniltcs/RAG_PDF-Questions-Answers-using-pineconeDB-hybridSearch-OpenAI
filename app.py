import streamlit as st
import os
from dotenv import load_dotenv

import time

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFDirectoryLoader

# 🔥 Pinecone imports
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_openai import ChatOpenAI


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# HuggingFace Embeddings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# LLM configuration
# -----------------------------
llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
            )
prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
            )


# ------------------------------------------------------
# CREATE VECTOR + SPARSE (BM25) INDEX IN PINECONE
# ------------------------------------------------------
def create_pinecone_retriever():
    if "retriever" not in st.session_state:

        # ---- Load PDFs ----
        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()

        # ---- Split text ----
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # ---- Initialize Pinecone ----
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        index_name = "hybrid-search-langchain-pinecone"

        # Create index if not exists
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,    # dimension of MiniLM-L6-v2 embeddings
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )

        pinecone_index = pc.Index(index_name)

        # ---- BM25 Sparse Encoder ----
        bm25_encoder = BM25Encoder().default()

        # ---- Create hybrid retriever ----
        retriever = PineconeHybridSearchRetriever(
            index=pinecone_index,
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
        )

        # Convert docs into plain text
        texts = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

        # ---- Store data in Pinecone (sparse + dense) ----
        retriever.add_texts(texts=texts, metadatas=metadatas)

        st.session_state.retriever = retriever


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="💻 Q&A + OpenAI + LangChain", layout="centered")
st.title("RAG Document Q&A With OpenAI + Pinecone Hybrid Search")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Build Pinecone Hybrid Index"):
    create_pinecone_retriever()
    st.success("Hybrid Vector Index (BM25 + Embeddings) is ready!")

if user_prompt and "retriever" in st.session_state:

    retriever = st.session_state.retriever
    ## relevant_docs = retriever.get_relevant_documents(user_prompt)

    # --- Build RAG Pipeline ---
    rag_chain = (
        {"context": RunnableLambda(lambda x: retriever.invoke(x["input"])), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time.process_time()
    response = rag_chain.invoke({"input": user_prompt})
    st.write(response)
    st.write(f"Response time: {time.process_time() - start:.2f} sec")
