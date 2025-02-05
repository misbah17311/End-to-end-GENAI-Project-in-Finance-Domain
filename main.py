import os
import streamlit as st
import pickle
import time
import langchain
from langchain.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
st.title("News research tool")
st.sidebar.title("News article URLs")
urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked=st.sidebar.button("Process URLs")
file_path="faiss_store_huggingface.pkl"
main_placefolder=st.empty()
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/openai-community/gpt2",  # Hugging Face model endpoint
    huggingfacehub_api_token="your_hugging_face_api_token",                  # Your Hugging Face API token
    temperature=0.7,  # Control randomness (moved outside model_kwargs)
             # Use nucleus sampling (moved outside model_kwargs)
    #model_kwargs={"max_length": 100}   # Other model parameters can go here if needed
    model_kwargs={"max_length": 256} ,
                         max_new_tokens=100
                         )
if process_url_clicked:
    #load data
    loader=UnstructuredURLLoader(urls=Urls)
    main_placefolder.text("data_loading_started...")

    #split data
    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        separators=['n\n', '\n', '-', ','],
        chunk_size=500
    )
    main_placefolder.text("text_splitter_started...")
    docs=text_splitter.split_documents(data)
    #create embeddings and save it to faiss index
    embeddings=HuggingFaceEmbeddings()
    vectorstore_openai= FAISS.from_documents(docs,embeddings)
    main_placefolder.text("vectorstore_embedding_started...")
    time.sleep(2)
    with open(file_path,"wb") as f:
       pickle.dump(vectorstore_openai,f)

query=main_placefolder.text("question")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore=pickl.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retrieval=vectorstore.as_retriever())
            result=chain({"question":query},return_only_outputs=True)
            #{"answer":"sources':[]}
            st.header("answer")
            st.subheader(result["answer"])

            #display sources if available
            sources=result.get("sources:","")
            if sources:
                st.subheader("sources:")
                sources_list=sources.split("\n")
                for source in sources_list:
                    st.write(source)
