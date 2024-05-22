import os
import getpass
import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyPDFLoader # for pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

# load groq and google api key from .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Groq AI Statical Book  Q & A")

# load groq model
llm = ChatGroq(groq_api_key=groq_api_key,model_name = "Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on provided context.
    Please provide most accurate response based on question.
    <context>                   
    {context}
    </context>
    Question: {input}                                                               
"""
)

def vector_embeddings():
    if "vectors" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFLoader("data/statical_book.pdf") # data ingestion                                                          )
            st.session_state.document = st.session_state.loader.load() # load document
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200) # Chunk Creation
            st.session_state.final_documents =  st.session_state.text_splitter.split_documents(st.session_state.document) # splitting docs
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents[:100], st.session_state.embeddings)# vector embeddings using GoogleGenerativeAIEmbeddings
prompt1 = st.text_input("Enter your question you wants to ask from statical learning book?")

if st.button("Ask"):
    vector_embeddings()
    st.write("Your db is ready")

# creating chainning
if prompt1:
    document_chain  = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    response = retriever_chain.invoke({"input": prompt1})
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
