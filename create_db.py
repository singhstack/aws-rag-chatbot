import json
import os
import sys
import boto3
import streamlit as st

#Using tital embedding models to generate embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

#Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.text import TextLoader


#Vector Embedding and Vector Store
from langchain.vectorstores import FAISS
#LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1",client = bedrock)

## Update Data ingestion for text files
def data_ingestion(data_folder):
    #loader=PyPDFDirectoryLoader("data")
    data_dir = os.path.join(os.getcwd(), data_folder)
    # List of documents
    documents = []
    # Loop through all files in the data directory (assuming .txt extension)
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            loader = TextLoader(file_path)
            document = loader.load()
            documents.append(document[0])
    print(len(documents))

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=100)
    
    docs=text_splitter.split_documents(documents)
    return docs

    
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index_check")


if __name__ == "__main__":
    print("starting")
    data_folder = "data"
    docs = data_ingestion(data_folder)
    get_vector_store(docs)
    print("Data Ingested and Vector Store Created")