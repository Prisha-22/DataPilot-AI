import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.agent.data_agent import run_agent


objective = """
Give me the total sales of the retail dataset and also explain
what data quality issues were found and how they were handled
during cleaning. Use exact analysis for the sales number and the
dataset knowledge base for the data-quality explanation.
"""


print("\n========== DATAPILOT AI AGENT TEST ==========")
print("\nManager Objective:")
print(objective)

result = run_agent(objective)

print("\n========== TASK PLAN ==========")
print(result["task_plan"])

print("\n========== TOOLS USED ==========")

for tool_call in result["tool_calls"]:
    print(
        f"- {tool_call['tool']}"
    )
    print(
        f"  Arguments: {tool_call['arguments']}"
    )

print("\n========== TOOL EVIDENCE ==========")

for tool_result in result["tool_results"]:

    print(f"\nTool: {tool_result['tool']}")

    print(
        f"Arguments: {tool_result['arguments']}"
    )

    print("Result:")

    print(
        tool_result["result"]
    )

print("\n========== RETRIEVED RAG SOURCES ==========")

if result["retrieved_sources"]:

    for source in result["retrieved_sources"]:

        print(
            f"\nSource ID: {source['id']}"
        )

        print(
            f"Similarity: {source['score']:.4f}"
        )

        print(
            f"Content:\n{source['text']}"
        )

else:

    print("No RAG sources were retrieved.")


print("\n========== FINAL AGENT ANSWER ==========")
print(result["answer"])