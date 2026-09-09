import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv




st.title("AI Chat App")

st.set_page_config(page_title="AI Chat App", page_icon=":speech_balloon:", layout="wide")

st.title("Welcome to My Chat App!")
st.markdown("This is a simple ai chat application built with Streamlit.")
load_dotenv()

with st.sidebar:
    st.header("Chat Settings")
    
    # api_key = st.text_input("OpenAI API Key", type="password")
    ai_model = st.selectbox("Select AI Provider", ["OpenAI", "Groq", "Google Gemini"], index=0)

    api_key = os.getenv("OPENAI_API_KEY")
    modelArray = []

    if ai_model == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")

        modelArray = ["gpt-4o",
        "gpt-4o-mini",
        "openai/gpt-oss-120b",
        "gpt-oss-120b",
        "gpt-oss-20b"]
    elif ai_model == "Groq":
        api_key = os.getenv("GROQ_API_KEY")
        modelArray = ["llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192"]
    elif ai_model == "Google Gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        modelArray = ["gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash"]

    model = st.selectbox("Select Model",modelArray, index=0)

    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource
def get_chain(api_key, model):
    if not api_key:
        return None

    provider = None
    
    if api_key.startswith("sk-proj-") or api_key.startswith("sk-"):
        provider = "openai"
    elif api_key.startswith("gsk_"):
        provider = "groq"
    elif api_key.startswith("AIzaSy"):
        provider = "google_genai"
    else:
        # अगर की का पैटर्न मैच न हो, तो मॉडल के नाम से अंदाजा लगाएं
        if "gemini" in model.lower():
            provider = "google_genai"
        elif "gpt" in model.lower():
            provider = "openai"
        elif "llama" in model.lower() or "mixtral" in model.lower():
            provider = "groq"
    
    # llm = ChatGroq(groq_api_key=api_key, model=model, temperature=0.7, streaming=True)
    llm = init_chat_model(
        provider + ":" + model,
        api_key=api_key,
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain


chain = get_chain(api_key, model)

if not chain:
    st.warning("Please enter your OpenAI API key to start chatting.")
else:
    for message in st.session_state["messages"]:
        if isinstance(message, HumanMessage):
            st.markdown(f"**You:** {message.content}")
        elif isinstance(message, AIMessage):
            st.markdown(f"**Assistant:** {message.content}")

    
    if question:= st.chat_input("Ask me anything..."):
        st.session_state["messages"].append(HumanMessage(content=question))
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream(input=question):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                st.session_state["messages"].append(AIMessage(content=full_response))

            except Exception as e:
                st.error(f"An error occurred: {e}")

