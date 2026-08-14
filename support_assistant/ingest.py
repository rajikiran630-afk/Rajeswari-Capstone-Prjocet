from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Paths
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=str(CHROMA_DIR))


# Recreate collection so the script can be safely run again
try:
    client.delete_collection("zepto_policies")
except Exception:
    pass

collection = client.create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


# Load all 8 documents
documents = []
ids = []
metadatas = []

for doc_path in sorted(DOCS_DIR.glob("doc_*.txt")):
    text = doc_path.read_text(encoding="utf-8").strip()

    if text:
        documents.append(text)
        ids.append(doc_path.stem)
        metadatas.append({"source": doc_path.name})

print(f"Loaded {len(documents)} documents.")


# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()


# Store documents + embeddings in ChromaDB
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


print("Ingestion completed successfully!")
print(f"ChromaDB collection: {collection.name}")
print(f"Documents stored: {collection.count()}")
print(f"Database location: {CHROMA_DIR}")