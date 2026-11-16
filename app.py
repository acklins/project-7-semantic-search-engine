#Streamlit interface for searcing documents stored in ChromaDB.

import streamlit as st

from search import search, get_collection_stats
from ingest import (
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_OVERLAP,
    ingest,
)

#Page configuration
st.set_page_config(
    page_title="Semantic Search Engine", page_icon="🔍",
    layout="wide",
)

#Cached search model

#Collection statistics

stats = get_collection_stats()

st.title("🔍 Semantic Search Engine")

st.write(
    "Search technical documentation using semantic similarity  "
    "powered by Sentence Transformers and ChromaDB."
)

# Sidebar configuration
with st.sidebar:
    st.header("Search Settings")

    st.metric("Documents Indexed", stats["unique_sources"])
    st.metric("Total Chunks", stats["total_chunks"])

    st.divider()

    n_results = st.slider(
        "Number of Results", 
        min_value=1, 
        max_value=10, 
        value=5,
    )

    use_threshold = st.checkbox(
        "Enable Distance Threshold", value=False,
    )

    distance_threshold = None

    if use_threshold:
        distance_threshold = st.slider(
            "Maximum Cosine Distance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            step=0.05,
        )

    st.divider()

    selected_sources = st.multiselect(
        "Filter by Source", 
        options=stats["source_names"],
        default=[],
        help="Leave empty to search all documents.",
    )

    st.divider()

    st.subheader("Re-index Documents")

    chunk_size = st.number_input(
        "Chunk Size (characters)",
        min_value=100, 
        max_value=2000, value=DEFAULT_CHUNK_SIZE,
        step=100,
    )

    overlap = st.number_input(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=DEFAULT_OVERLAP,
        step=25,
    )

    if st.button("Re-index Documents"):
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            with st.spinner("Re-indexing documents..."):
                try:
                    ingest(
                        chunk_size=int(chunk_size), overlap=int(overlap),
                    )
                    st.success("Documents re-indexed successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Re-indexing failed: {exc}")

# Search interface

st.subheader("Search Your Documents")

query = st.text_input(
    "Enter a question:",
    placeholder="How does FastAPI validate request data?",
)

if query.strip():
    with st.spinner("Searching documents..."):
        try:
            results = search(
                query=query,
                n_results=n_results,
                sources=selected_sources if selected_sources else None,
                distance_threshold=distance_threshold,
            )
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            results = []

    if not results:
        st.warning(
            "No matching documents found. "
            "Try a different query or increase the distance threshold. "
        )
    else:
        st.success(f"Found {len(results)} matching chunks.")

        for index, result in enumerate(results, start=1):
            score = result["score"]
            distance = result["distance"]

            if distance <= 0.3:
                relevance = "🟢 High relevance"
            elif distance <= 0.6:
                relevance = "🟡 Medium relevance"
            else:
                relevance = "🔴 Low relevance"

            with st.container(border=True):
                st.markdown(
                    f"### Result {index}: {result['source']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Similarity Score",
                        f"{score:.3f}",
                    )

                with col2:
                    st.metric(
                        "Cosine Distance",
                        f"{distance:.3f}",
                    )

                with col3:
                    st.write("**Relevance**")
                    st.write(relevance)

                st.caption(
                    f"Source: {result['source']} | " 
                    f"Chunk: {result['chunk_index']}"
                )

                st.write(result["text"])

else:
    st.info("Enter a question above to search your documents.")
