import argparse
from search import search

#Queries for the chunking experiment.
EVAL_SET = [
    {
        "query": "How do Python decorators and generators work?",
        "relevant_sources": ["python-advanced.txt"],
    },
    {
        "query": "How does FastAPI validate incoming request data?",
        "relevant_sources": ["fastapi.txt"],
    },
    {
        "query": "How do SQL JOIN operations combine database tables?",
        "relevant_sources": ["sql-databases.txt"],
    },
    {
        "query": "What is the difference between PUT and PATCH in REST APIs?",
        "relevant_sources": ["rest-apis.txt"],
    },
    {
        "query": "How do embeddings and cosine similarity support semantic search?",
        "relevant_sources": ["embeddings-and-vectors.txt"],
    },
]

#Compute source-level precision and recall.
def precision_recall(
    retrieved_sources: list[str], 
    relevant_sources: list[str],
) -> tuple[float, float]:
    
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)

    if not relevant:
        return 0.0, 0.0

    correct = len(retrieved.intersection(relevant))

    precision = correct / len(retrieved) if retrieved else 0.0
    recall = correct / len(relevant)

    return precision, recall

#Evaluate semantic search using gold-standard queries.
def evaluate(
        n_results: int = 5, 
        distance_threshold: float = None,
):
    
    if not EVAL_SET:
        print("Evaluation set is empty.")
        return 0.0, 0.0

    precisions = []
    recalls = []

    print("\n=== SEMANTIC SEARCH EVALUATION ===")
    print(f"Number of results: {n_results}")
    print(f"Distance threshold: {distance_threshold}")

    for index, entry in enumerate(EVAL_SET, start=1):
        query = entry["query"]
        relevant_sources = entry["relevant_sources"]

        results = search(
            query=query, 
            n_results=n_results, 
            distance_threshold=distance_threshold,
        )

        retrieved_sources = [
            result["source"] 
            for result in results
        ]

        precision, recall = precision_recall(
            retrieved_sources, 
            relevant_sources,
        )

        precisions.append(precision)
        recalls.append(recall)

        print(f"\nQuery {index}: {query}")
        print(f"Expected: {relevant_sources}")
        print(f"Retrieved: {sorted(set(retrieved_sources))}")
        print(f"Precision: {precision * 100:.1f}%")
        print(f"Recall: {recall * 100:.1f}%")

        print("Top results:")
        for rank, result in enumerate(results[:3], start=1):
            print(
                f"  {rank}. {result['source']} "
                f"(distance={result['distance']:.3f}, "
                f"score={result['score']:.3f})"
            )

    avg_precison = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    print("\n=== AVERAGE RESULTS ===")
    print(f"Average precision: {avg_precison * 100:.1f}%")
    print(f"Average recall: {avg_recall * 100:.1f}%")

    return avg_precison, avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--n-results", type=int,   default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    evaluate(n_results=args.n_results, distance_threshold=args.threshold)
