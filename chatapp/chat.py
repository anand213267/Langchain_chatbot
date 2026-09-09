"""
LangGraph RAG Demo — Streamlit UI
Run: uv run streamlit run chat.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from agent import MessagesState, agent
import chromadb
import pandas as pd

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangGraph RAG Demo",
    page_icon="🤖",
    layout="wide",
)

# ChromaDB का परसिस्टेंट पाथ सेट करें
CHROMA_PATH = "../chromaDB/chromaDB"

# ---------------------------------------------------------------------------
# NAVIGATION SIDEBAR (ऑप्शन मेनू)
# ---------------------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to Page:", ["💬 Chatbot Assistant", "📤 Upload Documents", "🗄️ ChromaDB Admin Panel"])

st.sidebar.divider()

# ---------------------------------------------------------------------------
# SCREEN 1: CHATBOT ASSISTANT
# ---------------------------------------------------------------------------
if page == "💬 Chatbot Assistant":
    st.title("🤖 LangGraph RAG Demo")
    st.caption("Calculator agent + knowledge base search — powered by LangGraph + Claude")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "tool_traces" not in st.session_state:
        st.session_state.tool_traces = []

    # Sidebar — tool trace
    with st.sidebar:
        st.header("🔧 Tool Trace")
        st.caption("Live view of which tools the agent called")

        if not st.session_state.tool_traces:
            st.info("No tool calls yet. Ask a question!")
        else:
            for i, trace in enumerate(reversed(st.session_state.tool_traces)):
                turn_label = f"Turn {len(st.session_state.tool_traces) - i}"
                with st.expander(turn_label, expanded=(i == 0)):
                    if not trace:
                        st.write("_No tools called — answered directly._")
                    else:
                        for call in trace:
                            st.markdown(f"**Tool:** `{call['name']}`")
                            st.markdown("**Input:**")
                            st.code(json.dumps(call["args"], indent=2), language="json")
                            st.markdown("**Output:**")
                            st.code(call["result"], language="text")
                            st.divider()

    # Chat display
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    # Input Prompt
    prompt = st.chat_input("Ask me to calculate something or explain an AI concept...")

    if st.button("Clear Chat History"):
        st.session_state.update({"chat_history": [], "tool_traces": []})

    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                initial_state: MessagesState = {
                    "messages": [HumanMessage(content=prompt)],
                    "llm_calls": 0,
                }
                result = agent.invoke(initial_state)

            pending_calls = {}
            messages = result["messages"]

            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        pending_calls[tc["id"]] = {
                            "name": tc["name"],
                            "args": tc["args"],
                            "result": "",
                        }
                elif isinstance(msg, ToolMessage):
                    if msg.tool_call_id in pending_calls:
                        pending_calls[msg.tool_call_id]["result"] = msg.content

            trace = list(pending_calls.values())
            st.session_state.tool_traces.append(trace)

            final_answer = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final_answer = msg.content
                    break

            st.markdown(final_answer)
            st.session_state.chat_history.append(("assistant", final_answer))

        st.rerun()

# ---------------------------------------------------------------------------
# SCREEN 2: UPLOAD DOCUMENTS (नया फीचर)
# ---------------------------------------------------------------------------
elif page == "📤 Upload Documents":
    st.title("📤 Upload Your Own Text Documents")
    st.caption("Upload a .txt file to automatically split it into chunks and generate embeddings for ChromaDB.")

    # 1. File Uploader widget
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

    if uploaded_file is not None:
        # फाइल का कंटेंट रीड करें
        text_content = uploaded_file.read().decode("utf-8")
        file_name = uploaded_file.name
        
        st.info(f"📄 Loaded file: **{file_name}** ({len(text_content)} characters)")
        
        # चंकिंग और प्रोसेसिंग बटन
        if st.button("⚡ Process & Add to Vector Database"):
            with st.spinner("Splitting text into chunks and creating embeddings..."):
                
                # 2. Text Splitting (Same as ingest.py logic)
                CHUNK_SIZE = 500
                paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
                
                chunks = []
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) + 2 <= CHUNK_SIZE:
                        current = (current + "\n\n" + para).strip()
                    else:
                        if current:
                            chunks.append(Document(page_content=current, metadata={"source": file_name}))
                        current = para
                if current:
                    chunks.append(Document(page_content=current, metadata={"source": file_name}))

                st.write(f"✅ Created **{len(chunks)}** chunks.")

                # 3. Generating Embeddings and Saving to ChromaDB
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                persistent_client = chromadb.PersistentClient(path=CHROMA_PATH)
                
                # 'my_collection' में डाक्यूमेंट्स अपेंड (add) हो जाएंगे
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    client=persistent_client,
                    collection_name="my_collection"
                )
                
                st.success("🎉 Embeddings successfully created and pushed to ChromaDB!")

# ---------------------------------------------------------------------------
# SCREEN 3: CHROMADB ADMIN PANEL
# ---------------------------------------------------------------------------
elif page == "🗄️ ChromaDB Admin Panel":
    st.title("🗄️ ChromaDB Admin Panel")
    st.caption("Browse collections, vectors, documents and metadata just like phpMyAdmin")
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()
        coll_names = [c.name for c in collections]

        if coll_names:
            selected_coll = st.sidebar.selectbox("Select Collection (Table):", coll_names)
            
            if selected_coll:
                st.subheader(f"📋 Table / Collection: `{selected_coll}`")
                
                collection = client.get_collection(selected_coll)
                data = collection.get()

                # print(data['ids'])

                if "clear_input" in st.session_state and st.session_state.clear_input:
                    st.session_state.delete_id = ""  # यहाँ इसे बदलने में कोई एरर नहीं आएगी
                    st.session_state.clear_input = False  # फ्लैग को वापस फॉल्स कर दें

                delete_id = st.text_input("Enter ID to delete", key="delete_id")

                if st.button("Delete Row"):
                    if delete_id:
                        if delete_id in data["ids"]:
                            collection.delete(ids=[delete_id])
                            st.success(f"Deleted row with ID: {delete_id}")
                            st.session_state.clear_input = True
                            st.rerun()
                        else:
                            st.error(f"ID '{delete_id}' not found in collection '{selected_coll}'.")



                if data["ids"]:
                    df = pd.DataFrame({
                        "ID": data["ids"],
                        "Document (Text Content)": data["documents"],
                        "Metadata (Tags)": [json.dumps(m, indent=2) for m in data["metadatas"]]
                    })

                    col1, col2 = st.columns(2)
                    col1.metric("Total Rows (Documents)", len(df))
                    col2.metric("Storage Status", "Active Connection")

                    st.write("---")
                    st.dataframe(df, use_container_width=True, height=500)
                else:
                    st.info(f"The collection '{selected_coll}' is empty.")
        else:
            st.sidebar.warning("No collections found in ChromaDB!")
            
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")