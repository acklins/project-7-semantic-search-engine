import argparse
from pathlib import Path
from tracemalloc import start
import chromadb
from sentence_transformers import SentenceTransformer
from streamlit import text
from torch import chunk

# ── Configuration ─────────────────
DOCS_DIR        = Path("docs")
CHROMA_PATH     = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME      = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP    = 100

# ── Utility Functions ─────────────
#Split text into fixed-size chunks with overlap.
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )
    
    if not text or not text.strip():
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks

#Read all .txt and .md files from docs_dir.
def load_documents(docs_dir: Path) -> list[dict]:

    if not docs_dir.exists() or not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir}"
        )

    documents = []

    files = sorted(
        list(docs_dir.glob("*.txt")) +
        list(docs_dir.glob("*.md"))
    )

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            print(f"Skipping empty file: {file_path}")
            continue
        documents.append({
            "filename": file_path.name,
            "text": text,
        })
    return documents

#Create (or retrieve) a persistent ChromaDB collection.
def get_collection(chroma_path: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(chroma_path))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    
    return collection

#Full ingestion pipeline: load → chunk → embed → upsert.
def ingest(
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    overlap: int = DEFAULT_OVERLAP
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )
    documents = load_documents(DOCS_DIR)

    if not documents:
        raise ValueError("No documents found to index.")

    chunk_ids = []
    chunk_texts = []
    chunk_metadata = []

    for document in documents:
        filename = document["filename"]
        chunks = chunk_text(document["text"], chunk_size, overlap)

        for index, text in enumerate(chunks):
            chunk_ids.append(f"{filename}::chunk_{index}")
            chunk_texts.append(text)
            
            chunk_metadata.append({
                "source": filename,
                "chunk_index": index,
                "chunk_size": chunk_size,
            })

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunk_texts)} chunks.")
    print(f"Chunk size: {chunk_size}, overlap: {overlap}")
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
    ).tolist()

    print("Saving embeddings to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    #Remove the previous index so different chunk sizes
    #are never mixed together during experiments.
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=chunk_metadata,
    )
    print("\n=== INGESTION COMPLETE ===")
    print(f"Documents indexed: {len(documents)}")
    print(f"Chunks stored: {collection.count()}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Storage location: {CHROMA_PATH.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index docs/ into ChromaDB")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap",    type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    ingest(chunk_size=args.chunk_size, overlap=args.overlap)
