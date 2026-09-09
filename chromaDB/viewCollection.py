import chromadb
import numpy as np

client = chromadb.PersistentClient(path="./chromaDB")
collection = client.get_collection(name="office_policies")

# data = collection.get(
#     ids=["doc_hr_leave_001", "doc_ops_timing_002"]
# )

data = collection.get(
    include=["documents", "embeddings"],
    ids=["doc_hr_leave_001", "doc_ops_timing_002", "doc_fin_expense_003", "doc_it_secure_004", "doc_hr_wfh_005"]
)
# print("Retrieved data:", data)

# for doc_id, doc_content, doc_embedding in zip(data["ids"], data["documents"], data["embeddings"]):
#     print(f"Document ID: {doc_id}")
#     print(f"Content: {doc_content}")
#     print(f"Embedding: {doc_embedding[:10]}...")  # Print only the first 10 dimensions for brevity
#     print("-" * 50)

def cosine_similarity(vec1, vec2):
    # Convert inputs to numpy arrays just in case
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Calculate dot product
    dot_product = np.dot(a, b)
    
    # Calculate L2 norms (magnitudes)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid Division by Zero errors
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

embedding_1 = data["embeddings"][0]  # Embedding for doc_hr_leave_001
embedding_2 = data["embeddings"][1]  # Embedding for doc_ops_timing_002
embedding_3 = data["embeddings"][2]  # Embedding for doc_ops_timing_003
embedding_4 = data["embeddings"][3]  # Embedding for doc_ops_timing_004
embedding_5 = data["embeddings"][4]  # Embedding for doc_ops_timing_005


similarity = cosine_similarity(embedding_1, embedding_2)
print(f"Cosine Similarity (Doc 1 vs Doc 2): {similarity:.4f}")
similarity = cosine_similarity(embedding_1, embedding_3)
print(f"Cosine Similarity (Doc 1 vs Doc 3): {similarity:.4f}")
similarity = cosine_similarity(embedding_1, embedding_4)
print(f"Cosine Similarity (Doc 1 vs Doc 4): {similarity:.4f}")
similarity = cosine_similarity(embedding_1, embedding_5)
print(f"Cosine Similarity (Doc 1 vs Doc 5): {similarity:.4f}")