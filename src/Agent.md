# Agent.py Code Explanation

## Overview
`Agent.py` implements an intelligent agent that can answer user questions about a data warehouse. It supports both SQL-based queries for structured data and Retrieval-Augmented Generation (RAG) for unstructured data, automatically choosing the best strategy based on the user's input.

## Key Components

### 1. Imports and Setup
- Loads environment variables and initializes required modules.
- Imports utility functions for database access, schema formatting, vector search, and LLM initialization.

### 2. State Definition
- Uses Python `TypedDict` to define the `State` object, which tracks the current question, query, result, answer, history, table list, query type, and RAG answer.

### 3. LLM Initialization
- The agent uses a helper (`get_llm`) to initialize the language model (OpenAI or AWS Bedrock) based on environment variables.

### 4. Prompt Templates
- Defines prompt templates for generating SQL queries and answers, using LangChain's `ChatPromptTemplate` and `MessagesPlaceholder` for conversational context.

### 5. Schema and Vector Store
- Loads table and column metadata from the database.
- Builds a vector store for semantic search over table descriptions.

### 6. Utility Functions
- `ensure_str_list`: Ensures the history is always a list of strings.

### 7. Workflow Steps
- **route_query**: Determines if the user's question should be answered with SQL or RAG, using a prompt and structured output parsing.
- **select_tables_llm**: Uses the LLM to select relevant tables for SQL queries.
- **generate_query**: Generates a syntactically correct SQL query using the LLM and schema context.
- **execute_query**: Executes the SQL query against the database and returns results.
- **generate_answer**: Converts SQL results into a natural language answer.
- **query_knowledge_base**: Uses AWS Bedrock Knowledge Base to answer questions that require RAG.

### 8. Workflow Graph
- Uses LangGraph's `StateGraph` to define the workflow.
- Entry point is `route_query`, which conditionally routes to either the SQL or RAG branch.
- SQL branch: `select_tables_llm` → `generate_query` → `execute_query` → `generate_answer` → END
- RAG branch: `query_knowledge_base` → END

### 9. Interactive CLI
- Provides a command-line interface for users to ask questions.
- The agent automatically routes queries and prints answers, indicating whether the response came from SQL or RAG.

## How It Works
1. User asks a question.
2. The agent decides if the question fits a SQL query or requires RAG.
3. If SQL, it selects relevant tables, generates and executes a query, and returns a natural language answer.
4. If RAG, it queries the AWS Knowledge Base and returns the generated answer, including citations if available.

## Extensibility
- The agent is modular and can be extended to support more query types, additional data sources, or custom routing logic.

## Requirements
- Environment variables for database and AWS configuration.
- AWS Bedrock Knowledge Base must be set up for RAG functionality.

---
For further details, see the inline comments and docstrings in `Agent.py`.
