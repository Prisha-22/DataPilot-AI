import json
from pathlib import Path

import numpy as np

from src.rag.knowledge_base import build_dataset_knowledge
from src.rag.embeddings import (
    create_document_embeddings,
    create_query_embedding,
)


INDEX_PATH = Path(
    "data/processed/rag_index.json"
)


def build_rag_index():
    """
    Build and save the vector index for the DataPilot
    knowledge base using batched embeddings.
    """

    knowledge = build_dataset_knowledge()

    texts = [
        chunk["text"]
        for chunk in knowledge
    ]

    titles = [
        chunk["id"]
        for chunk in knowledge
    ]

    print(
        f"Creating {len(knowledge)} embeddings in one batch..."
    )

    embeddings = create_document_embeddings(
        texts=texts,
        titles=titles
    )

    if len(embeddings) != len(knowledge):
        raise ValueError(
            "Number of embeddings does not match "
            "number of knowledge chunks."
        )

    index = []

    for chunk, embedding in zip(
        knowledge,
        embeddings
    ):

        index.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "embedding": embedding
        })

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        INDEX_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            index,
            file
        )

    print(
        f"\nRAG index saved to: {INDEX_PATH}"
    )

    return index

def load_rag_index():
    """
    Load the saved vector index.
    """

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "RAG index not found. "
            "Run build_rag_index() first."
        )

    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def search_knowledge(
    query,
    top_k=3,
    min_score=0.65
):
    """
    Find the most semantically relevant knowledge chunks.

    Results below the minimum similarity threshold are removed
    so weakly related knowledge is not presented as evidence.
    """

    index = load_rag_index()

    query_embedding = create_query_embedding(
        query
    )

    results = []

    for item in index:

        score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        if score >= min_score:

            results.append({
                "id": item["id"],
                "text": item["text"],
                "score": score
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]