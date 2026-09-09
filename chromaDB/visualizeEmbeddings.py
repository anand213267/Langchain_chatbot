import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import chromadb

client = chromadb.PersistentClient(path="./chromaDB")
collection = client.get_collection(name="office_policies")

data = collection.get(include=["embeddings", "documents", "metadatas"])

embeddings = data["embeddings"]
documents = data["documents"]
metadatas = data["metadatas"]

# Convert embeddings list to a numpy array
embeddings_array = np.array(embeddings)

# 2. Reduce dimensions from High-Dim (e.g., 768 or 10) down to 2D (X, Y)
# PCA is fast and excellent for basic structured visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_array)

# 3. Plotting the graph
plt.figure(figsize=(10, 8))

# Extract X and Y coordinates
x_coords = embeddings_2d[:, 0]
y_coords = embeddings_2d[:, 1]

# Plot each document point
scatter = plt.scatter(x_coords, y_coords, s=100, cmap='Set1')

# 4. Annotate each point with its Document content or ID
for i, doc in enumerate(documents):
    # Slice the document string if it's too long for the label
    label = doc if len(doc) < 40 else doc[:40] + "..."
    plt.annotate(
        label, 
        (x_coords[i], y_coords[i]), 
        xytext=(5, 5), 
        textcoords='offset points', 
        fontsize=9
    )

plt.title("ChromaDB Document Embeddings Visualization (2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()