import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.rag.rag_qa import answer_from_rag


query = """
What data quality issues were found in the retail dataset,
and how were they handled during cleaning?
"""


print("\n========== DATAPILOT RAG Q&A TEST ==========")

print(f"\nUser Question:\n{query}")

result = answer_from_rag(
    query,
    top_k=3
)

print("\n========== RAG ANSWER ==========")
print(result["answer"])

print("\n========== RETRIEVED SOURCES ==========")

for source in result["sources"]:
    print(
        f"{source['id']} "
        f"(similarity: {source['score']:.4f})"
    )