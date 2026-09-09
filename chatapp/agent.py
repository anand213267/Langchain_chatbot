import operator
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
import chromadb
from langchain_chroma import Chroma

load_dotenv()

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides a by b.

    Args:
        a: First int
        b: Second int
    """
    return a / b



@tool
def search_docs(query: str) -> str:
    """Search the knowledge base for information about AI/ML concepts, LangGraph, RAG, 
    or custom developer profiles (ID, wife name, coding language, age), and secret projects.

    Args:
        query: The search query describing what information you need
    """
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # vectorstore = FAISS.load_local(
    #     index_path, embeddings, allow_dangerous_deserialization=True
    # )
    persistent_client = chromadb.PersistentClient(path='../chromaDB/chromaDB')
    
    vectorstore = Chroma(
        client=persistent_client,
        collection_name="my_collection",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


tools = [add, multiply, divide, search_docs]
tools_by_name = {t.name: t for t in tools}



model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
model_with_tools = model.bind_tools(tools)

def llm_call(state: MessagesState) -> dict:
    """LLM decides whether to call a tool or return a final answer."""
    strict_system_prompt = (
        "You are a strict Retrieval-Augmented Generation (RAG) assistant.\n\n"
        "CRITICAL RULE:\n"
        "1. For ANY question regarding projects, hidden valleys, secret stories, "
        "engineers, custom lore, timers, or files (like Chronos, Aether-Delta, drones, "
        "clocks, locks, or unexpected scenarios), you MUST call the `search_docs` tool first. "
        "DO NOT guess or make up a story.\n"
        "2. Use math tools for calculations.\n"
        "3. Only answer directly if it is a generic greeting (like 'hi' or 'hello')."
    )

    response = model_with_tools.invoke(
        [SystemMessage(content=strict_system_prompt)] + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def tool_node(state: MessagesState) -> dict:
    """Execute every tool call requested by the LLM."""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": results}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()


load_dotenv()