import streamlit as st
import chromadb
import pandas as pd

st.title("ChromaDB Admin Panel (phpMyAdmin Style)")

# ChromaDB से कनेक्ट करें
client = chromadb.PersistentClient(path="./chromaDB")
collections = client.list_collections()

# ड्रॉपडाउन से कलेक्शन चुनें (जैसे phpMyAdmin में टेबल चुनते हैं)
coll_names = [c.name for c in collections]
selected_coll = st.selectbox("Select Collection (Table):", coll_names)

if selected_coll:
    collection = client.get_collection(selected_coll)
    data = collection.get()
    
    # डेटा को टेबल फॉर्मेट में सेट करें
    df = pd.DataFrame({
        "ID": data["ids"],
        "Document": data["documents"],
        "Metadata": [str(m) for m in data["metadatas"]]
    })
    
    # Streamlit की इंटरैक्टिव टेबल में डेटा दिखाएं
    st.dataframe(df, use_container_width=True)