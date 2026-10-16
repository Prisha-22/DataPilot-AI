import json

from src.llm.gemini_client import client
from src.rag.retriever import search_knowledge

from src.tools.analysis_tools import (
    get_total_sales,
    get_sales_by_category,
    get_sales_by_region,
    get_sales_trend,
    get_high_risk_inventory,
    get_demand_forecast,
    get_forecast_evaluation,
)


MODEL_NAME = "gemini-3.5-flash-lite"
PLANNER_MODEL = "gemini-3.5-flash-lite"

SYSTEM_INSTRUCTION = """
You are DataPilot AI, an AI decision-support agent for retail
sales and inventory management.

Your job is to understand the manager's objective, select the
appropriate analysis tools, analyze the actual uploaded retail
dataset, and provide evidence-based findings.

STRICT RULES:

1. Use tools whenever numerical or dataset-specific information
   is required.

2. Never invent numbers, facts, causes, or business conditions.

3. Use only evidence returned by the available tools.

4. Never assume information that is not present in the dataset.
   For example, do not assume warehouse stock, supplier lead time,
   operating hours, delivery time, purchase orders, or budget.

5. Inventory coverage is expressed in days only. Do not convert
   coverage into hours or claim a specific stock-out time.

6. When making a recommendation, clearly connect it to observed
   evidence from the dataset.

7. If the available dataset is insufficient to support a
   recommendation, explicitly say what information is missing.

8. When using a time window such as 7 days or 30 days, clearly
   state the time window used in the answer.

9. Do not confuse correlation or observation with causation.
   Describe sales trends as observed changes, not as proof of future
   growth or demand.
   
10. All monetary values in this retail dataset are in Indian
    Rupees (INR). Always use ₹ and never use $, USD, or another
    currency symbol.

11. Keep the response understandable to a retail manager.

12. Prefer factual findings first, followed by cautious
    recommendations supported by those findings.

13. For inventory risk, use the definitions from the analysis tool:
    High Stock-Out Risk means inventory coverage is less than 3 days.
    Medium Stock-Out Risk means coverage is from 3 days to less than 7 days.
    Potential Overstock means coverage is 30 days or more.
    Never change or reinterpret these thresholds.
14. Mean Absolute Error (MAE) is an error metric, not an accuracy
    percentage. Report MAE as forecast error and do not convert it
    into an accuracy percentage unless a separately defined metric
    supports that conversion.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_total_sales",
        "description": "Calculate total sales across the entire retail dataset.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "get_sales_by_category",
        "description": "Calculate total sales for each product category.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "get_sales_by_region",
        "description": "Calculate total sales for each geographic region.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "get_sales_trend",
        "description": "Compare the most recent 7 days of sales with the previous 7 days.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "get_high_risk_inventory",
        "description": "Identify store-product combinations with high stock-out risk based on recent demand.",
        "parameters": {
            "type": "object",
            "properties": {
                "recent_days": {
                    "type": "integer",
                    "description": "Number of recent days used to calculate demand.",
                }
            },
            "required": ["recent_days"],
        },
    },
    {
        "type": "function",
        "name": "get_demand_forecast",
        "description": "Forecast future demand for a selected store and product.",
        "parameters": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "string",
                    "description": "Store ID such as S001.",
                },
                "product_id": {
                    "type": "string",
                    "description": "Product ID such as P0001.",
                },
                "window": {
                    "type": "integer",
                    "description": (
                        "Number of recent historical days used for the moving "
                        "average. Must be an integer such as 7. Never use words "
                        "such as 'historical', 'recent', or 'default'."
                    ),
                },
                "forecast_days": {
                    "type": "integer",
                    "description": "Number of future days to forecast.",
                },
            },
            "required": [
                "store_id",
                "product_id",
                "window",
                "forecast_days",
            ],
        },
    },
        {
        "type": "function",
        "name": "get_forecast_evaluation",
        "description": (
            "Evaluate the accuracy of the moving-average demand "
            "forecast for a selected store and product using "
            "historical data. Returns Mean Absolute Error (MAE)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "store_id": {
                    "type": "string",
                    "description": "Store ID such as S002.",
                },
                "product_id": {
                    "type": "string",
                    "description": "Product ID such as P0007.",
                },
                "window": {
                    "type": "integer",
                    "description": (
                        "Number of recent historical days used for evaluation. "
                        "Must be an integer such as 7. Never use words such as "
                        "'historical' or 'recent'."
                    ),
                },
            },
            "required": [
                "store_id",
                "product_id",
                "window",
            ],
        },
    },
        {
        "type": "function",
        "name": "search_dataset_knowledge",
        "description": (
            "Search the DataPilot dataset knowledge base for "
            "relevant contextual information about the uploaded "
            "retail dataset. Use this for semantic or descriptive "
            "questions where retrieved knowledge is useful."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The question or topic to search for "
                        "in the dataset knowledge base."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of relevant knowledge chunks "
                        "to retrieve."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]

def create_task_plan(manager_objective):
    """
    Create a task-specific analysis plan from the manager's objective.
    """

    planner_prompt = f"""
Create a task-specific analysis plan for DataPilot AI.

Manager objective:
{manager_objective}

Available analysis tools:
- get_total_sales
- get_sales_by_category
- get_sales_by_region
- get_sales_trend
- get_high_risk_inventory
- get_demand_forecast
- get_forecast_evaluation
- search_dataset_knowledge

Instructions:
1. Understand exactly what the manager is asking.
2. Create only the minimum number of steps needed to answer the objective.
3. Do not add unrelated or unnecessary analyses.
   However, include a tool when the manager's question depends on
   that information, uses it as a business premise, asks "why",
   or requires combining multiple factors to answer correctly.
4. Do not create a forecast step unless the manager explicitly asks
   for forecasting or future demand.
5. Use get_sales_trend when the objective:
   - explicitly asks about sales trend, growth, increase, decrease,
     recent performance, or change over time, OR
   - uses a sales trend as part of the reasoning for the decision.

6. For the "Sales performance" objective, use:
   - get_total_sales
   - get_sales_by_category
   - get_sales_by_region
   when the objective asks for an overall sales performance
   analysis or asks to identify the most important sales pattern.

7. Use get_high_risk_inventory for stock-out risk, restocking,
   replenishment, inventory urgency, or "which items should I
   prioritize" questions.
8. Use search_dataset_knowledge only when dataset definitions,
   contextual information, or semantic knowledge is needed.
9. A task plan must only contain tools that can actually be executed
   with information available from the manager objective.
10. Each step must contain exactly one relevant tool.
11. Prefer 1–2 tools for a focused objective, unless a specific
    objective rule requires additional tools to answer correctly.
    12. Do not calculate or invent any results.
13. Return only the task-specific plan.
14. Use the exact tool parameter names when describing a plan.
15. Use get_forecast_evaluation when the manager asks about
    forecast accuracy, reliability, error, or performance.
16. For get_demand_forecast, use:
    store_id, product_id, window, forecast_days.
17. For get_forecast_evaluation, use:
    store_id, product_id, window.
18. Whenever a tool parameter has type integer, provide an actual
    integer value. Never substitute descriptive words for numeric
    parameters.
19. For the current moving-average forecasting workflow, use
    window=7 unless the manager explicitly requests another window.
"""

    interaction = client.interactions.create(
        model=PLANNER_MODEL,
        input=planner_prompt
    )

    return interaction.output_text


def execute_tool(name, arguments):
    """
    Execute the tool selected by Gemini.
    """

    arguments = arguments or {}

    if name == "get_total_sales":
        result = get_total_sales()

    elif name == "get_sales_by_category":
        result = get_sales_by_category()

    elif name == "get_sales_by_region":
        result = get_sales_by_region()

    elif name == "get_sales_trend":
        result = get_sales_trend()

    elif name == "get_high_risk_inventory":
        recent_days = arguments.get("recent_days", 30)

        result = get_high_risk_inventory(
            recent_days=recent_days
        )

        total_high_risk = len(result)
    
        top_results = (
            result.head(20)
            .reset_index()
            .to_dict(orient="records")
        )

        result = {
            "recent_days": recent_days,
            "total_high_risk_combinations": total_high_risk,
            "top_20_highest_risk": top_results
        }

    elif name == "get_demand_forecast":
        result = get_demand_forecast(
            store_id=arguments.get("store_id"),
            product_id=arguments.get("product_id"),
            window=arguments.get("window", 7),
            forecast_days=arguments.get("forecast_days", 7),
        )
    elif name == "get_forecast_evaluation":
        result = get_forecast_evaluation(
            store_id=arguments.get("store_id"),
            product_id=arguments.get("product_id"),
            window=arguments.get("window", 7),
        )

    elif name == "search_dataset_knowledge":
        query = arguments.get("query")

        if not query:
            raise ValueError(
                "A query is required for knowledge search."
            )

        top_k = arguments.get("top_k", 3)

        return search_knowledge(
            query=query,
            top_k=top_k
        )

    else:
        raise ValueError(f"Unknown tool: {name}")

    if hasattr(result, "reset_index"):
        result = result.reset_index()

    if hasattr(result, "to_dict"):
        result = result.to_dict(orient="records")

    return result


def run_agent(manager_objective):
    """
    Run the DataPilot AI agent for a manager's objective.

    Returns:
        dict containing:
        - manager_objective
        - task_plan
        - tool_calls
        - answer
    """

    # ---------------------------------------------------------
    # Step 1: Create task-specific plan
    # ---------------------------------------------------------

    task_plan = create_task_plan(
        manager_objective
    )

    print("\n========== AI TASK PLAN ==========")
    print(task_plan)

    # ---------------------------------------------------------
    # Step 2: Give the plan to the execution agent
    # ---------------------------------------------------------

    execution_prompt = f"""
Manager objective:
{manager_objective}

Task plan created by the planner:
{task_plan}

Execute this task plan using the available tools.

Execute only the tools necessary to answer the manager objective.

Do not execute optional tools merely to add extra information.

Use:
- analysis tools for exact numerical calculations
- search_dataset_knowledge for semantic or descriptive dataset knowledge

For inventory coverage:
- Report coverage only in days.
- Never convert days into hours.
- Never calculate or state a specific clock time until stock-out.
- Do not reinterpret the inventory-risk thresholds.

Use actual dataset results for numerical findings.
Do not invent information.
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=execution_prompt,
        system_instruction=SYSTEM_INSTRUCTION,
        tools=TOOLS,
    )

    # ---------------------------------------------------------
    # Step 3: Track tools selected by the Agent
    # ---------------------------------------------------------

    tool_calls = []
    tool_results = []
    retrieved_sources = []
    # ---------------------------------------------------------
    # Step 4: Execute selected tools
    # ---------------------------------------------------------

    while True:

        function_calls = [
            step
            for step in interaction.steps
            if step.type == "function_call"
        ]

        # -----------------------------------------------------
        # No more tools → Agent has finished
        # -----------------------------------------------------

        if not function_calls:

            final_answer = interaction.output_text

            return {
                "manager_objective": manager_objective,
                "task_plan": task_plan,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "retrieved_sources": retrieved_sources,
                "answer": final_answer
            }

        function_results = []

        for step in function_calls:

            print(
                f"\nAgent selected tool: {step.name}"
            )

            print(
                f"Tool arguments: {step.arguments}"
            )

            tool_calls.append({
                "tool": step.name,
                "arguments": step.arguments
            })

            result = execute_tool(
                step.name,
                step.arguments
            )

            tool_results.append({
                "tool": step.name,
                "arguments": step.arguments,
                "result": result
            })

            if step.name == "search_dataset_knowledge":
                for source in result:

                    retrieved_sources.append({
                        "id": source["id"],
                        "score": source["score"],
                        "text": source["text"]
                    })

            function_results.append(
                {
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                result,
                                default=str
                            ),
                        }
                    ],
                }
            )

        # -----------------------------------------------------
        # Step 5: Continue the interaction with tool results
        # -----------------------------------------------------

        interaction = client.interactions.create(
            model=MODEL_NAME,
            previous_interaction_id=interaction.id,
            input=function_results,
            system_instruction=SYSTEM_INSTRUCTION,
            tools=TOOLS,
        )