import streamlit as st
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
from langchain.tools import WikipediaQueryRun
load_dotenv()
import arxiv
# Initialize LLM
from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-4o")

# AgentState TypedDict
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# Helper to create retrieval tools from text files
def make_retrieval_tool_from_text(file, name, desc):
    docs = TextLoader(file, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = vs.as_retriever()

    def tool_func(query: str) -> str:
        results = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in results])
    return Tool(name=name, description=desc, func=tool_func)

# Tools
wiki_tool = Tool(
    name="Wikipedia",
    description="Wikipedia tool to fetch general knowledge from Wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

def arxiv_search(query: str) -> str:
        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = [f"{r.title}\n{r.summary[:1000]}" for r in search.results()]
        return "\n\n".join(results) if results else "No Papers found"

arxiv_tool = Tool(
    name="ArXiv",
    description="ArXiv tool to fetch papers from ArXiv",
    func=arxiv_search
)

internal_tool1 = make_retrieval_tool_from_text(
    "research_notes.txt", "Internal_research-notes", "Search internal research notes"
)
internal_tool2 = make_retrieval_tool_from_text(
    "sample_docs.txt", "Sample_research_notes", "Search internal samples"
)

tools = [wiki_tool, arxiv_tool, internal_tool1, internal_tool2]

# Create agent
react_node = create_react_agent(llm, tools)
builder = StateGraph(AgentState)
builder.add_node("agentic_rag", react_node)
builder.set_entry_point("agentic_rag")
builder.add_edge("agentic_rag", END)
graph = builder.compile()

# Streamlit UI
st.title("Agentic RAG Search")

user_query = st.text_area("Enter your query:")

if st.button("Search"):
    if user_query.strip() == "":
        st.warning("Please enter a query first.")
    else:
        state = {"messages": [HumanMessage(content=user_query)]}
        result = graph.invoke(state)
        st.subheader("Agent Response:")
        st.write(result["messages"][-1].content)
