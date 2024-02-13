import os
import streamlit as st
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI

from dotenv import load_dotenv
load_dotenv() #take environment variable from .env

st.title('News Research tool')
st.sidebar.title('News Article URLs')

# creating a list of urls***********************************************
urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i}")
    urls.append(url)

process_url_clicked= st.sidebar.button("Process URLs")

file_path="faiss_store_openai.pkl"

# the urls are passed into the unstructuredURLloader*********************
main_placeholder= st.empty()
if process_url_clicked:
    # loading the data
    loaders=UnstructuredURLLoader(urls=urls)
    main_placeholder.text('Data loading started.......')
    data = loaders.load()

# recursive character text splitter**************************************
    main_placeholder.text('Data splitting in progress.......')
    text_splitter= RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000,
        chunk_overlap=200
        )
    docs= text_splitter.split_documents(data)

    #Create embeddings and save it to FAISS index
    main_placeholder.text('Embedding Vector started Building.......')

    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text('Data embedded.......')

    # saving the index
    vectorstore_openai.save_local("faiss_index")


query=main_placeholder.text_input("Qestion:")


llm= OpenAI(temperature=0.9, max_tokens=500)
if query:
    if os.path.exists(file_path):
        vector_openai_index = FAISS.load_local("faiss_index", embeddings)
        chain= RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever= vectorindex_openai.as_retriever())
        result= chain({"question": query}, return_only_outputs=True)
        st.header('Answer')
        st.subheader(result["answer"])