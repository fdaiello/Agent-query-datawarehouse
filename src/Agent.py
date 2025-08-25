from typing import TypedDict, Annotated, List, Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
load_dotenv()
from llm_utils import get_llm
from db_utils_redshift import get_columns, get_tables, query_database, get_schema_comment, DB_PLATFORM, DB_SPECIFICS
from schema_vector import create_vectorstore, search_vectorstore
from schema_format import format_schema_description
from aws_kb_utils import retrieve_and_generate, format_citations
from typing import cast

# Utility to ensure history is always List[str]
def ensure_str_list(history) -> list[str]:
    return [str(h) for h in history if isinstance(h, (str, int, float, bool))]

# Define the application state with memory (history)

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    history: List[str]
    table_list: List[str]
    query_type: Literal["sql", "rag"]
    rag_answer: str


llm = get_llm()

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Define the prompt for generating SQL queries, including history
system_message ="""You are a helpful assistant.
Generate syntactically correct SQL queries based on the user's question.
Target database: {db_platform}. {db_specifics}
Unless the user specifies in their question a specific number of examples they wish to obtain, always limit your query to at most 10 results.
Pay attention to use only the column names that you can see in the schema description.
Never query for all the columns from a specific table, only ask for a few relevant columns given the question.
Domain specific knowledge:{schema_comments}
Database schema: {db_schema_str}"""
user_prompt = "Question: {input}"
query_prompt_template = ChatPromptTemplate(
    [
        ("system", system_message),
        MessagesPlaceholder("history"),
        ("user", user_prompt)
    ]
)

class TableSubsetOutput(TypedDict):
    tables: Annotated[str, ..., "Subset of schema_info with only relevant tables for the query."]

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

class QueryRouterOutput(TypedDict):
    query_type: Literal["sql", "rag"]

# Fetch schema_info and build vector store once at startup
TABLE_INFO = get_tables()
SCHEMA_COMMENTS = get_schema_comment()
TABLE_VECTORSTORE = create_vectorstore(TABLE_INFO)
COLUMNS_INFO = get_columns()

# Step 0: Route the query to either SQL or RAG
def route_query(state: State) -> State:
    """Route the query to either SQL or RAG based on the user's question and available tables."""
    question = state["question"]
    table_list_str = "\n".join([f"{t['table_name']}: {t['table_comment']}" for t in TABLE_INFO])
    
    prompt = ChatPromptTemplate([
        ("system", f"You are an expert in determining if a user's question can be answered by querying a SQL database or if it requires information retrieval from a knowledge base (RAG). Given the user's question and the available database tables with their descriptions, decide if the question can be answered by SQL. If the question can be answered by SQL, respond with 'sql'. Otherwise, respond with 'rag'. Available database tables:\n{table_list_str}"),
        ("user", f"Question: {question}")
    ])
    prompt_value = prompt.invoke({})
    structured_llm = llm.with_structured_output(QueryRouterOutput)
    result = structured_llm.invoke(prompt_value)
    result = cast(QueryRouterOutput, result)
    return {
        **state,
        "query_type": result["query_type"]
        }

# Step 1 (Vector Search): use vector search to select relevant table
def select_tables_vector(state: State) -> State:
    """Select relevant tables from schema_info for the user's question using vector search."""
    history: list[str] = ensure_str_list(state.get("history", []))
    relevant_subset = search_vectorstore(state["question"], TABLE_VECTORSTORE, top_k=5)
    new_history: list[str] = history + [f"User: {state['question']}", f"Relevant tables: {relevant_subset}"]
    return {
        **state,
        "question": state["question"],
        "query": state["query"],
        "result": state["result"],
        "answer": state["answer"],
        "history": ensure_str_list(new_history),
        "table_list": relevant_subset
    }

# Step 1: Use LLM to select relevant tables from TABLE_INFO
def select_tables_llm(state: State) -> State:
    """Call LLM to decide which tables should be used for the user's question."""
    history: list[str] = ensure_str_list(state.get("history", []))
    # Prepare table info string for LLM
    table_list_str = "\n".join([f"{t['table_name']}: {t['table_comment']}" for t in TABLE_INFO])
    prompt = ChatPromptTemplate([
        ("system", "Given the user's question and the list of tables with descriptions, return a comma separated list of table names that are relevant for answering the question in order of relevance."),
        ("user", f"Question: {state['question']}\nTables:\n{table_list_str}")
    ])
    prompt_value = prompt.invoke({})
    result = llm.invoke(prompt_value).content
    # Parse result as comma-separated list of table names
    relevant_subset: List[str] = []
    if isinstance(result, str):
        relevant_subset = [t.strip() for t in result.split(",") if t.strip()]
    new_history: list[str] = history + [f"User: {state['question']}", f"Relevant tables: {relevant_subset}"]
    return {
        "question": state["question"],
        "query": state.get("query", ""),
        "result": state.get("result", ""),
        "answer": state.get("answer", ""),
        "history": ensure_str_list(new_history),
        "table_list": relevant_subset,
        "query_type": state.get("query_type", "sql"),
        "rag_answer": state.get("rag_answer", "")
    }

# Step 2: Generate SQL query using schema subset
def generate_query(state: State) -> State:
    """Generate SQL query to fetch information using schema subset."""
    history: list[str] = ensure_str_list(state.get("history", []))
    table_list = state.get("table_list", [])
    table_list_comments = [t for t in TABLE_INFO if t.get('table_name') in table_list]
    db_schema_str = format_schema_description(table_list_comments, COLUMNS_INFO)

    # Convert table_list list to string for prompt
    prompt_value = query_prompt_template.invoke(
        {
            "input": state["question"],
            "history": history,
            "db_schema_str": db_schema_str,
            "db_platform": DB_PLATFORM,
            "db_specifics": DB_SPECIFICS,
            "schema_comments": SCHEMA_COMMENTS
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt_value)
    result = cast(QueryOutput, result)
    new_history: list[str] = history + [f"User: {state['question']}", f"SQL: {result['query']}"]
    return {
        **state,
        "question": state["question"],
        "query": result["query"],
        "result": state.get("result", ""),
        "answer": state.get("answer", ""),
        "history": ensure_str_list(new_history),
        "table_list": table_list,
        "query_type": state.get("query_type", "sql"),
        "rag_answer": state.get("rag_answer", "")
    }

# Function to execute the SQL query
def execute_query(state: State) -> State:
    """Execute the SQL query using Redshift Data API and return the result."""
    result = query_database(state["query"])
    result_str = str(result)
    history: list[str] = ensure_str_list(state.get("history", []))
    new_history: list[str] = history + [f"SQL: {state['query']}", f"Result: {result_str}"]
    return {
        "question": state["question"],
        "query": state["query"],
        "result": result_str,
        "answer": state.get("answer", ""),
        "history": ensure_str_list(new_history),
        "table_list": state.get("table_list", []),
        "query_type": state.get("query_type", "sql"),
        "rag_answer": state.get("rag_answer", "")
    }

# Define the prompt for generating the answer, including history
answer_system_message = """You are a helpful AI assistant. Given the user's question and the SQL query result, \
provide a natural language answer. If the query result is empty, state that no information was found.\
"""
answer_user_prompt = "Question: {question}\nSQL Result: {result}"
answer_prompt_template = ChatPromptTemplate(
    [
        ("system", answer_system_message),
        MessagesPlaceholder("history"),
        ("user", answer_user_prompt)
    ]
)

# Function to generate the answer
def generate_answer(state: State) -> State:
    """Generate a natural language answer based on the question and query result."""
    history: list[str] = ensure_str_list(state.get("history", []))
    prompt_value = answer_prompt_template.invoke(
        {
            "question": state["question"],
            "result": state["result"],
            "history": history
        }
    )
    answer = llm.invoke(prompt_value).content
    new_history: list[str] = history + [f"Answer: {answer}"]
    return {
        "question": str(state["question"]),
        "query": str(state.get("query", "")),
        "result": str(state.get("result", "")),
        "answer": str(answer),
        "history": ensure_str_list(new_history),
        "table_list": state.get("table_list", []),
        "query_type": state.get("query_type", "sql"),
        "rag_answer": state.get("rag_answer", "")
    }

# RAG Branch Functions
def query_knowledge_base(state: State) -> State:
    """Query the AWS Knowledge Base for information."""
    question = state["question"]
    history: list[str] = ensure_str_list(state.get("history", []))
    
    try:
        # Use retrieve_and_generate for a complete RAG response
        rag_result = retrieve_and_generate(question)
        
        rag_answer = rag_result.get('answer', 'No answer found.')
        citations = rag_result.get('citations', [])
        
        # Format citations if available
        formatted_citations = format_citations(citations) if citations else ""
        
        # Combine answer with citations
        full_rag_answer = rag_answer
        if formatted_citations:
            full_rag_answer += f"\n\nSources:\n{formatted_citations}"
        
        new_history: list[str] = history + [f"User: {question}", f"RAG Answer: {full_rag_answer}"]
        
        return {
            "question": state["question"],
            "query": state.get("query", ""),
            "result": state.get("result", ""),
            "answer": full_rag_answer,
            "history": ensure_str_list(new_history),
            "table_list": state.get("table_list", []),
            "query_type": state.get("query_type", "rag"),
            "rag_answer": full_rag_answer
        }
        
    except Exception as e:
        error_message = f"Error querying knowledge base: {str(e)}"
        new_history: list[str] = history + [f"User: {question}", f"Error: {error_message}"]
        
        return {
            "question": state["question"],
            "query": state.get("query", ""),
            "result": state.get("result", ""),
            "answer": error_message,
            "history": ensure_str_list(new_history),
            "table_list": state.get("table_list", []),
            "query_type": state.get("query_type", "rag"),
            "rag_answer": error_message
        }

# Build the LangGraph workflow with conditional routing
workflow = StateGraph(State)

# Add all nodes
workflow.add_node("route_query", route_query)
workflow.add_node("select_tables", select_tables_llm)
workflow.add_node("generate_query", generate_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("query_knowledge_base", query_knowledge_base)

# Set entry point
workflow.set_entry_point("route_query")

# Define conditional routing function
def decide_path(state: State) -> str:
    """Decide whether to go to SQL or RAG path based on query_type."""
    query_type = state.get("query_type", "sql")
    if query_type == "rag":
        return "query_knowledge_base"
    else:
        return "select_tables"

# Add conditional edge from route_query
workflow.add_conditional_edges(
    "route_query",
    decide_path,
    {
        "select_tables": "select_tables",
        "query_knowledge_base": "query_knowledge_base"
    }
)

# SQL path edges
workflow.add_edge("select_tables", "generate_query")
workflow.add_edge("generate_query", "execute_query")
workflow.add_edge("execute_query", "generate_answer")
workflow.add_edge("generate_answer", END)

# RAG path edge
workflow.add_edge("query_knowledge_base", END)

app = workflow.compile()


# Example usage
if __name__ == "__main__":
    # Interactive loop for user queries
    print("\n💬 Ask me questions about your database or general knowledge! (type 'exit' or 'quit' to stop)")
    print("🔄 The system will automatically route your query to either SQL database or Knowledge Base.")
    history: list[str] = []
    while True:
        user_input = input("\n❓ Your question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        # Always cast history to list[str]
        safe_history: list[str] = ensure_str_list(history)
        state: State = {
            "question": user_input,
            "query": "",
            "result": "",
            "answer": "",
            "history": safe_history,
            "table_list": [],
            "query_type": "sql",  # Default, will be determined by route_query
            "rag_answer": ""
        }
        try:
            result = app.invoke(state)
            query_type = result.get('query_type', 'unknown')
            if query_type == "sql":
                print(f"\n🗄️  [SQL Query] Answer: {result['answer']}")
            elif query_type == "rag":
                print(f"\n📚 [Knowledge Base] Answer: {result['answer']}")
            else:
                print(f"\n📊 Answer: {result['answer']}")
            # Update history for next turn, always as list[str]
            history = ensure_str_list(result["history"])
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different question.")