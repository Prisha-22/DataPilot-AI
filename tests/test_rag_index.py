import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.rag.retriever import build_rag_index


print("\n========== DATAPILOT RAG INDEX TEST ==========")

index = build_rag_index()

print(f"\nTotal indexed chunks: {len(index)}")

print("\nFirst indexed chunk:")
print(f"ID: {index[0]['id']}")
print(f"Embedding dimensions: {len(index[0]['embedding'])}")

print("\nRAG index generation completed successfully.")