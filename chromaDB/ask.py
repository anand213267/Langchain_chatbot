import chromadb

client = chromadb.PersistentClient(path="./chromaDB")
collection = client.get_collection(name="office_policies")

# Retrieve all documents in the collection
result = collection.query(
    query_texts=["Can I work from my house two days a week?"],
    n_results=3
)

# Print the retrieved documents and their metadata
for i, doc in enumerate(result['documents'][0]):
    print(f"Document {i + 1}:")
    print(f"Content: {doc}")
    print(f"Metadata: {result['metadatas'][0][i]}")
    print("-" * 50)
