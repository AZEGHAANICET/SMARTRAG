import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from typing import List, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from pydantic import BaseModel

load_dotenv()


llm = init_chat_model("openai:gpt-4o")

docs = TextLoader("research_notes.txt", encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter()
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()



class RAGCoTState(BaseModel):
    question:str
    sub_steps: List[str]=[]
    retrieved_docs: List[Document]=[]
    answer: str=""

def plan_steps(state: RAGCoTState)->RAGCoTState:
    prompt =f"Break the question into 2 - 3 reasoning steps: \n\n {state.question}"
    result = llm.invoke(prompt).content
    subqs = [line.strip("- ") for line in result.split("\n") if line.strip()]
    return state.model_copy(update={"sub_steps": subqs})

def retrieve_per_step(state: RAGCoTState)->RAGCoTState:
    all_docs = []

    for sub in state.sub_steps:
        docs= retriever.invoke(sub)
        all_docs.extend(docs)
    return state.model_copy(update={"retrieved_docs": all_docs})


def generate_answer(state: RAGCoTState)->RAGCoTState:
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt=f"""
    You are answering a complex queston using reasoning and retrieved documents.
    
    Question: {state.question}
    
    Relevant information:
    {context}
    
    Now synthetize a well-reasoned final answer.
    """
    result = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer": result})


builder = StateGraph(RAGCoTState)
builder.add_node("planner", plan_steps)
builder.add_node("retriever", retrieve_per_step)
builder.add_node("responder", generate_answer)

builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()

if __name__ == "__main__":
    query="What are the additional experiments in transformer evaluation?"
    state = RAGCoTState(question=query)
    final = graph.invoke(state)
    print("Reasoning steps : ", final["sub_steps"])
    print("Final Answer : ", final["answer"])