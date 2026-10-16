import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.rag.retriever import search_knowledge


query = "What data quality issues were found and how were they handled during cleaning?"

print("\n========== DATAPILOT SEMANTIC RETRIEVAL TEST ==========")

print(f"\nQuery: {query}")

results = search_knowledge(
    query,
    top_k=3
)

print("\n--- RETRIEVED KNOWLEDGE ---")

for rank, result in enumerate(results, start=1):

    print(f"\nRank {rank}")
    print(f"ID: {result['id']}")
    print(f"Similarity Score: {result['score']:.4f}")
    print(f"Content:\n{result['text']}")

print("\nSemantic retrieval completed successfully.")