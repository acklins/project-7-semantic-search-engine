from pathlib import Path
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_PATH, COLLECTION_NAME, MODEL_NAME

#Keep this section?
# ── Configuration (must match ingest.py) ──────────
CHROMA_PATH     = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME      = "all-MiniLM-L6-v2"


#Load and cache the sentence embeddign model.
@lru_cache(maxsize=1)
def get_embeddings_model():
    return SentenceTransformer(MODEL_NAME)

#Connect to the existing persistent ChromaDB collection.
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name=COLLECTION_NAME)

#Search indexed documents and return matching chunks.
def search(
    query: str,
    n_results: int = 5,
    sources: list[str] = None,
    distance_threshold: float = None,
) -> list[dict]:

    if not query or not query.strip():
        return []

    if n_results <= 0:
        return []

    if sources is not None and not sources:
        return []

    try:
        collection = get_collection()
    except chromadb.errors.NotFoundError:
        return[]

    if collection.count() == 0:
        return []

    where_filter = None

    if sources is not None:
        where_filter = {"source": {"$in": sources}}

    model = get_embeddings_model()
    
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where_filter,
        include=["documents","metadatas", "distances"],
    )
    ranked_results = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if (
            distance_threshold is not None 
            and distance > distance_threshold
        ):
            continue

        ranked_results.append({
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance":float(distance),
                "score":float(1 - distance),
        })

    return ranked_results

#Return collection stats for the Streamlit interface.
def get_collection_stats() -> dict:

    try:
        collection = get_collection()
    except chromadb.errors.NotFoundError:
        return {
            "total_chunks": 0, "unique_sources": 0, "source_names": []
        }

    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "total_chunks": 0, "unique_sources": 0, "source_names": [],
        }

    records = collection.get(include=["metadatas"])

    source_names = sorted({
        metadata["source"] 
        for metadata in records["metadatas"]
    })
    return {
        "total_chunks": total_chunks, "unique_sources": len(source_names), "source_names": list(source_names)
    }
