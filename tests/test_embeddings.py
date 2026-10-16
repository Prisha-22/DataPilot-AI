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
from src.rag.embeddings import create_embedding


print("\n========== DATAPILOT EMBEDDING TEST ==========")

knowledge = build_dataset_knowledge()

first_chunk = knowledge[0]

print(f"\nChunk ID: {first_chunk['id']}")

embedding = create_embedding(
    first_chunk["text"]
)

print(f"Embedding type: {type(embedding)}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")

print("\nEmbedding generation completed successfully.")