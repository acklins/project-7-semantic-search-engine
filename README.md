# Module 7 Project — Semantic Search Engine (Starter)

## Overview

Build a semantic search tool over a document collection. The tool loads, chunks,
embeds, and stores documents in ChromaDB, then provides a Streamlit interface
where users can type a question and see ranked results. You'll also run a
chunking experiment and document your findings.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Index the document corpus:
   ```bash
   python ingest.py
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit search interface — your primary work file |
| `ingest.py` | Document loading, chunking, and ChromaDB storage |
| `search.py` | Search functions: query ChromaDB and rank results |
| `evaluate.py` | Evaluation framework: precision and recall |
| `docs/` | Starter document corpus (add your own documents here) |
| `chroma_data/` | Persistent ChromaDB storage (created on first run) |
| `requirements.txt` | Python dependencies |

## Requirements

See the project brief on the course platform for the full rubric. Key sections:

1. **Document Ingestion** — load `docs/` folder, configurable chunk size/overlap,
   store in ChromaDB with source and chunk-index metadata
2. **Semantic Search** — ranked results with scores, configurable `n_results` and
   threshold, edge-case handling
3. **Streamlit Interface** — search input, sidebar controls (filter, re-index,
   result count), at least one `st.metric()`
4. **Chunking Experiment** — re-index at 2+ chunk sizes, run 5 identical queries,
   compare and document findings in this README
5. **Code Quality** — docstrings, modular files, working `requirements.txt`

## Chunking Experiment Results
Chunk Size Experiment

Objective

Evaluate how document chunk size affects semantic search retrieval quality using Sentence Transformers and ChromaDB.

Experiment Setup

The same eight technical documents and five evaluation queries were used for both configurations.

Configuration		Chunk Size		Overlap			Indexed Chunks
A				      500 characters		100 characters				53
B				      200 characters		50 characters				137

Both configurations used the all-MiniLM-L6-v2 embedding model, cosine distance, and a maximum of five retrieved chunks per query. No distance threshold was applied.

Evaluation Results

Metric							      500 Characters			200 Characters
Average source-level precision		90.0%				   90.0%
Average source-level recall			100.0%				100.0%
Top-1 source accuracy				   100.0%				100.0%
Indexed chunks						      53					   137

Top-Ranked Similarity Scores

Evaluation Query					      500 Characters		200 Characters
Python decorators and generators		   0.553					0.606
FastAPI request validation				   0.678					0.698
SQL JOIN operations					      0.518					0.520
REST API PUT vs. PATCH				      0.683					0.598
Embeddings and cosine similarity		   0.638					0.649
Average							            0.614					0.614

Similarity score is calculated as 1 - cosine distance.

Findings

1.Both configurations achieved 100% recall. Every evaluation query retrieved its expected source document.

2. Both configurations achieved 90% average precision. The Python decorators and generators query also retrieved python-fundamentals.txt, reducing precision for that query to 50%. The remaining four queries achieved 100% precision.

3. Smaller chunks did not improve source-level retrieval accuracy. Precision and recall were identical across both configurations.

4. The 200-character configuration produced more chunks. It generated 137 chunks compared with 53 for the 500-character configuration, increasing the number of stored embeddings by approximately 158%.

5. Top-ranked similarity scores were mixed. The 200-character configuration had higher scores for four queries, while the 500-character configuration had a higher score for the REST API query. Their average top-ranked scores were effectively equal.

Conclusion

Both chunk sizes performed equally well on the five source-level evaluation queries. The 500-character configuration is selected as the default because it achieves the same precision and recall while requiring fewer stored embeddings and retaining more surrounding text in each chunk.

This experiment demonstrates that smaller chunks do not necessarily improve semantic search accuracy. Chunk size should be selected based on retrieval quality, context requirements, and storage considerations.

The evaluation was limited to five queries and source-level relevance labels. Further testing with more queries and passage-level relevance judgments would provide stronger evidence.

Reproducing the Experiment

# Configuration A
python ingest.py --chunk-size 500 --overlap 100
python evaluate.py --n-results 5 > results_500.txt

# Configuration B
python ingest.py --chunk-size 200 --overlap 50
python evaluate.py --n-results 5 > results_200.txt

The evaluation output files contain the top three ranked results for each query, including source filenames, cosine distances, and similarity scores.

