# DataPilot AI

## Explainable, Task-Aware Decision Support for Retail Sales & Inventory Management

🚀 [Live Demo](https://datapilot-ai-22.streamlit.app/) · [GitHub Repository](https://github.com/Prisha-22/DataPilot-AI)

DataPilot AI is an AI-powered retail analytics and decision-support application that allows a manager to upload a retail dataset, describe a business objective in natural language, and receive an evidence-based analysis.

The system combines traditional data analytics, task-aware AI agents, semantic retrieval (RAG), demand forecasting, and interactive visualization.

---

## Key Features

### 1. Dataset Upload

Supports:

* CSV
* Excel (`.xlsx`)

The uploaded dataset becomes the active dataset used by the analysis tools and RAG knowledge pipeline.

### 2. Dataset Profiling

The system reports:

* Number of rows
* Number of columns
* Column names
* Data types
* Missing values
* Unique values
* Duplicate rows

### 3. Data Cleaning

The cleaning pipeline performs dataset-specific preprocessing and quality handling.

Examples include:

* Converting the `Date` column to datetime
* Detecting invalid negative demand forecasts
* Replacing invalid negative demand forecast values with `0`
* Checking missing values
* Checking duplicate rows

### 4. Task-Aware AI Agent

The manager can select a predefined objective or enter a custom natural-language objective.

The Agent:

1. Understands the manager's request
2. Creates a task-specific plan
3. Selects the required analysis tools
4. Executes those tools on the active dataset
5. Retrieves relevant knowledge when required
6. Synthesizes the evidence into a manager-friendly answer

### 5. Analysis Tools

The application currently supports:

* Total sales analysis
* Sales by product category
* Sales by region
* Sales trend analysis
* High stock-out risk identification
* Demand forecasting
* Forecast evaluation using Mean Absolute Error (MAE)

### 6. Agentic RAG

DataPilot AI uses a dataset-specific knowledge base for semantic questions.

The RAG pipeline is:

```text
Active Dataset
      ↓
Knowledge Generation
      ↓
Gemini Embeddings
      ↓
Vector Index
      ↓
Semantic Retrieval
      ↓
Relevant Dataset Knowledge
      ↓
Agent
```

The Agent decides when knowledge retrieval is appropriate.

Exact numerical questions continue to use structured analysis tools rather than relying on semantic similarity.

### 7. Demand Forecasting

The current forecasting workflow uses a moving-average baseline.

Example:

```text
Store: S002
Product: P0007
Window: 7 days
Forecast horizon: 7 days
```

The system also evaluates the historical forecasting method using Mean Absolute Error (MAE).

MAE is reported as a forecast-error metric and is not interpreted as an accuracy percentage.

### 8. Explainability

The application exposes:

* AI task plan
* Tools selected by the Agent
* Tool evidence
* Retrieved RAG knowledge
* Executive KPIs
* Visual insights
* Final answer

This allows the user to understand how the final answer was produced.

---

## Example Manager Objectives

### Sales Performance

> Give me an overall sales performance assessment, including total sales, sales trend, and top-performing product category.

### Inventory Risk

> Which store-product combinations currently have the highest stock-out risk? Explain the evidence.

### Data Quality

> What data quality issues were found in the dataset and how were they handled during cleaning?

### Demand Forecasting

> Forecast demand for Store S002 and Product P0007 for the next 7 days and tell me how accurate the forecasting method is based on historical data.

### Business Health

> Give me an overall business health assessment covering sales performance, sales trend, category performance, and inventory risk.

---

## System Architecture

```text
                     MANAGER
                        │
                        ▼
                 DATASET UPLOAD
                        │
                        ▼
              PROFILING + CLEANING
                        │
                        ▼
                 ACTIVE DATASET
                        │
                        ▼
              MANAGER OBJECTIVE
                        │
                        ▼
                 TASK PLANNER
                        │
                        ▼
                TASK-AWARE AGENT
                 /      │       \
                /       │        \
               ▼        ▼         ▼
        ANALYSIS     RAG        FORECAST
          TOOLS      TOOLS       TOOLS
               \       │        /
                \      │       /
                 ▼     ▼      ▼
                    EVIDENCE
                       │
                       ▼
               AGENT SYNTHESIS
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       KPIs          CHARTS      SOURCES
          └────────────┼────────────┘
                       ▼
                 FINAL ANSWER
```

---

## Project Structure

```text
DataPilot-AI/
│
├── app/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── retail_store_inventory.csv
│   │
│   └── processed/
│       └── cleaned_retail_data.csv
│
├── src/
│   ├── agent/
│   │   └── data_agent.py
│   │
│   ├── analysis/
│   │   ├── demand_analysis.py
│   │   ├── inventory_analysis.py
│   │   ├── product_analysis.py
│   │   ├── sales_analysis.py
│   │   └── trend_analysis.py
│   │
│   ├── cleaning/
│   │   ├── clean.py
│   │   └── quality.py
│   │
│   ├── forecasting/
│   │   └── demand_forecast.py
│   │
│   ├── llm/
│   │   └── gemini_client.py
│   │
│   ├── profiling/
│   │   └── profile.py
│   │
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── knowledge_base.py
│   │   ├── rag_qa.py
│   │   └── retriever.py
│   │
│   └── tools/
│       └── analysis_tools.py
│
├── tests/
│   ├── test_agent.py
│   ├── test_analysis_tools.py
│   ├── test_clean.py
│   ├── test_demand_analysis.py
│   ├── test_embeddings.py
│   ├── test_forecast.py
│   ├── test_inventory_analysis.py
│   ├── test_knowledge_base.py
│   ├── test_profile.py
│   ├── test_quality.py
│   ├── test_rag_index.py
│   ├── test_rag_qa.py
│   ├── test_retriever.py
│   ├── test_sales_analysis.py
│   └── test_trend_analysis.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

Generated runtime files such as the active dataset and RAG vector index are intentionally excluded from version control.

---

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file and provide the required Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit the `.env` file.

---

## Running the Application

From the project root:

```powershell
streamlit run app/main.py
```

The application will open in the browser.

---

## Testing

Individual components can be tested through the files in the `tests/` directory.

Examples:

```powershell
python tests/test_agent.py
```

```powershell
python tests/test_retriever.py
```

```powershell
python tests/test_rag_index.py
```

```powershell
python tests/test_forecast.py
```

---

## Current Dataset Demonstration

The demonstration retail dataset contains:

* 73,100 rows
* 15 columns
* 5 stores
* 20 products
* 5 product categories
* 4 regions
* Date range from 2022-01-01 to 2024-01-01

Example observed results from the demonstration dataset include:

* Total sales: ₹550,228,884.91
* Recent 7-day sales change: +10.64%
* Top sales category: Furniture
* High stock-out risk combinations: 86

These values are dataset-specific and should be recalculated whenever a different dataset is uploaded.

---

## Limitations

The current system does not include:

* Supplier lead times
* Warehouse inventory
* Purchase orders
* Delivery schedules
* Store operating constraints
* Supplier-specific replenishment rules

Therefore, the system provides analytical decision support rather than automatically placing or scheduling replenishment orders.

The current demand forecasting method is a simple moving-average baseline and can be replaced by more advanced forecasting models in future versions.

---

## Future Scope

Potential extensions include:

* Multi-agent architecture
* Advanced forecasting models
* Automated reorder-point optimization
* Supplier and warehouse integration
* Scenario analysis
* Role-based dashboards
* Persistent vector databases
* More advanced anomaly detection
* Human approval workflows for recommendations

---

## Technology Stack

* Python
* Pandas
* NumPy
* Streamlit
* Plotly
* Google Gemini
* Gemini Embeddings
* RAG / Semantic Retrieval
* OpenPyXL
* Python-dotenv
