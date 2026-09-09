import os
import glob
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
import chromadb

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "testdocs")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_and_split():
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
            """Load .txt files and split into ~500-char chunks with overlap."""
            chunks = []
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            # print(text)
            # print(os.path.basename(path))

            source = os.path.basename(path)

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 <= CHUNK_SIZE:
                    current = (current + "\n\n" + para).strip()
                else:
                    if current:
                        chunks.append(Document(page_content=current, metadata={"source": source}))
                    # If single paragraph is already > CHUNK_SIZE, keep it as-is
                    current = para

            if current:
                chunks.append(Document(page_content=current, metadata={"source": source}))

    return chunks

def main():
    print("Loading and splitting documents...")
    chunks = load_and_split()
    print(f"  Split into {len(chunks)} chunks")

    print("Embedding and building FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # vectorstore = FAISS.from_documents(chunks, embeddings)
    persistent_client = chromadb.PersistentClient(path="../chromaDB/chromaDB")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings, # <--- यहाँ हमने अपना कस्टम दिमाग दे दिया
        client=persistent_client,         # <--- यहाँ हमने अपना Persistent Client दे दिया
        collection_name="my_collection"
    )

    print("Data successfully saved using ChromaDB Persistent Client!")
    # vectorstore.save_local(INDEX_DIR)

    # print(f"Done! Index saved to {INDEX_DIR}/")
    # print(f"  Total chunks indexed: {len(chunks)}")

if __name__ == "__main__":
    main()