from src.llm.gemini_client import client
from src.rag.retriever import search_knowledge


MODEL_NAME = "gemini-3.5-flash-lite"

def answer_from_rag(query, top_k=3):
    """
    Answer a semantic question using retrieved
    DataPilot knowledge.
    """

    retrieved = search_knowledge(
        query,
        top_k=top_k
    )

    context_parts = []

    for result in retrieved:
        context_parts.append(
            f"""
SOURCE: {result["id"]}

{result["text"]}
""".strip()
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the user's question using only the retrieved
DataPilot knowledge below.

User question:
{query}

Retrieved knowledge:
{context}

Rules:
1. Use only information explicitly stated in the retrieved
   knowledge.
2. Do not infer, assume, or add facts that are not explicitly
   present in the retrieved knowledge.
3. Do not describe an operation as fixing an "issue" unless
   the source explicitly identifies it as an issue.
4. If the retrieved knowledge does not contain enough
   information, clearly say that the available knowledge
   is insufficient.
5. Do not use semantic similarity scores as factual evidence.
6. When mentioning a fact, identify the relevant source ID.
7. Preserve numerical values exactly as provided by the source.
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt
    )

    return {
        "answer": interaction.output_text,
        "sources": [
            {
                "id": result["id"],
                "score": result["score"]
            }
            for result in retrieved
        ]
    }