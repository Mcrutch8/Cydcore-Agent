# agent.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage

def initialize_model():
    
    """Initialize the QA model and return the QA chain."""
    # Load environment variables from a .env file
    load_dotenv(override=True)
    openai_key = os.getenv("openai_key")

    # Initialize the loader with the directory path containing the PDFs
    loader = PyPDFDirectoryLoader("/Users/maxcrutchfield/Desktop/Cydcor_Proj/data")

    # Load all PDFs in the directory
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Initialize embeddings and language model
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    llm = ChatOpenAI(model="gpt-4", api_key=openai_key)

    # Create vector store
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Define system prompts
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Create prompt templates
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a customer support representative for the company Cydcor. "
                    "Please answer only Cydcor-related questions. Answer the user's question "
                    "based ONLY on the below context:\n\n{context}"
                ),
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create chains
    retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever_chain, document_chain)

    return qa_chain

def get_answer(qa_chain, query, chat_history):
    """Process the user's query and return the answer."""
    result = qa_chain.invoke({"input": query, "chat_history": chat_history})
    return result["answer"]
