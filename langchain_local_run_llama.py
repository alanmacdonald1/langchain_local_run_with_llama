# main.py (FastAPI)
import logging
from fastapi import BackgroundTasks
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from datetime import datetime

app = FastAPI()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow only Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import bs4
import time

import os
from langchain import hub
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatLlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatLlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from typing import List, Union

import re

# Langchain imports

from langchain.agents import AgentType, initialize_agent

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# LLM wrapper
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
# Conversational memory
from langchain.memory import ConversationBufferWindowMemory
# Embeddings and vectorstore
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.agents import initialize_agent, Tool, AgentExecutor

from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent

from langchain.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, UnstructuredPowerPointLoader

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.tools import Tool
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import (
    ConversationBufferMemory
)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.document_loaders import PyPDFLoader

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

# Global variables
STORE = {}

# Define global variables for the retriever, LLM, and other key components
agent = None
retriever = None
llm = None
initialized = False
prompt_template = None


# Custom InMemoryHistory class without inheritance conflict
class InMemoryHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

    def get_messages(self):
        return self.messages


def get_session_history(user_id: str, conversation_id: str) -> InMemoryHistory:
    """Get or create session history."""
    if (user_id, conversation_id) not in STORE:
        STORE[(user_id, conversation_id)] = InMemoryHistory()
    return STORE[(user_id, conversation_id)]


def load_documents(directory):
    """Helper function to load documents from a given directory."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, file))
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(os.path.join(directory, file))
        elif file.endswith('.csv'):
            loader = CSVLoader(os.path.join(directory, file))
        elif file.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(os.path.join(directory, file))
        else:
            continue
        documents.extend(loader.load())
    return documents


# print("sleeping")
# time.sleep(60)

def LOAD_DATA_SOURCES_AND_MODEL():
    global agent, retriever, prompt_template, llm, initialized

    # If already initialized, return immediately
    if initialized:
        return

    # Fetch the OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Load documents from multiple directories
    support_docs = load_documents("docs_to_query/support")
    hr_docs = load_documents("docs_to_query/HR")
    people_docs = load_documents("docs_to_query/people")

    # Combine all documents
    all_documents = support_docs + hr_docs + people_docs

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(all_documents)

    # Store documents in a Chroma vector database
    vector_store = Chroma.from_documents(
        documents=split_documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        collection_name="combined"
    )

    print("Documents loaded and vector store created.")

    chat_history = ConversationBufferMemory(output_key='answer', context_key='context', memory_key='chat_history',
                                            return_messages=True)

    # Define the model (local Llama-2)
    llm = ChatLlamaCpp(
        model_path="chat_model/openhermes-2.5-mistral-7b.Q5_K_M.gguf",
        n_gpu_layers=100,
        n_batch=4042,
        n_ctx=5048,
        # temperature=0.01,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True
    )

    # Create retriever
    retriever = vector_store.as_retriever()

    # Define prompt template
    # Define the prompt template with dynami

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are Sammy, the AI chatbot, here to assist users with queries about IT, programming languages, holidays, promotions, and more. Your goal is to provide accurate, helpful, and professional answers.

First, carefully think about the user's question and the context provided. Then, think again to ensure that your answer is correct, detailed, and clear. Make sure that your response is free from ambiguity or errors.

If applicable, correct any syntax errors in your language and aim to provide the most efficient and optimized solution where necessary.

Context:
{context}

Question: {question}

Please provide a thoughtful, detailed, and concise answer:
Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the chain
    rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
    )

    agent = rag_chain

    initialized = True


def AI_query(QUESTION):
    global agent

    if agent is None:
        raise Exception("Agent not initialized. Please run LOAD_DATA_SOURCES_AND_MODEL first.")

    print("Asking user query...")

    try:
        # Manage the tool invocations
        result = agent.invoke(QUESTION)
        print(result)
        # Ensure the result is in string format
        if isinstance(result, dict):
            # If the result is already in dictionary form, return it directly
            return result
        elif isinstance(result, (list, tuple)):
            # If it's a list or tuple, convert it to string or handle it accordingly
            return str(result)
        else:
            # If it's a string or other type, return as is
            return result
    except Exception as e:
        return str(e)


@app.on_event("startup")
async def startup_event():
    print("hi")
    LOAD_DATA_SOURCES_AND_MODEL()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    data: str


@app.post("/ai-query", response_model=AnswerResponse)
async def ai_query(request: QuestionRequest):
    question = request.question
    try:
        result = AI_query(question)
        answer = str(result)
        # TT= datetime.now()
        # answer = f"The time is ${TT}"
        print(f"sending ${answer}")
        return {"data": answer}
    except Exception as e:
        return {"error": str(e)}
