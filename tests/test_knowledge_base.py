import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.rag.knowledge_base import build_dataset_knowledge


print("\n========== DATAPILOT RAG KNOWLEDGE TEST ==========")

knowledge = build_dataset_knowledge()

print(f"\nTotal knowledge chunks: {len(knowledge)}")

for i, chunk in enumerate(knowledge[:10], start=1):
    print(f"\n--- CHUNK {i} ---")
    print(f"ID: {chunk['id']}")
    print(chunk["text"][:500])

print("\nKnowledge base generation completed successfully.")