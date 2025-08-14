from typing import TypedDict, Annotated, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
load_dotenv()
from db_utils import get_columns, get_tables, query_redshift, get_schema_comment, REDSHIFT_SCHEMA
from schema_vector import create_vectorstore, search_vectorstore
from schema_format import format_schema_description

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

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Define the prompt for generating SQL queries, including history
system_message = """You are a helpful assistant.
Generate syntactically correct Redshift SQL queries based on the user's question.
Unless the user specifies in their question a specific number of examples they wish to obtain, always limit your query to at most 10 results.
Database has several schemas. Always work on the schema {schema}.
Pay attention to use only the column names that you can see in the schema description.
Never query for all the columns from a specific table, only ask for a few relevant columns given the question.
Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
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

# Fetch schema_info and build vectorstore once at startup
TABLE_INFO = get_tables(REDSHIFT_SCHEMA)
SCHEMA_COMMENTS = get_schema_comment(REDSHIFT_SCHEMA)
TABLE_VECTORSTORE = create_vectorstore(TABLE_INFO)
COLUMNS_INFO = get_columns(REDSHIFT_SCHEMA)

# Step 1 (Vector Search): use vector search to select relevant table
def select_tables_vector(state: State) -> State:
    """Select relevant tables from schema_info for the user's question using vector search."""
    history: list[str] = ensure_str_list(state.get("history", []))
    relevant_subset = search_vectorstore(state["question"], TABLE_VECTORSTORE, top_k=5)
    new_history: list[str] = history + [f"User: {state['question']}", f"Relevant tables: {relevant_subset}"]
    return {
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
        ("system", "You are a helpful assistant. Given the user's question and the list of tables with descriptions, return a comma separated list of table names that are relevant for answering the question. Only include table names, no comments or extra text."),
        ("user", f"Question: {state['question']}\nTables:\n{table_list_str}")
    ])
    result = llm.invoke(prompt.invoke({})).content
    # Parse result as comma-separated list of table names
    relevant_subset: List[str] = []
    if isinstance(result, str):
        relevant_subset = [t.strip() for t in result.split(",") if t.strip()]
    new_history: list[str] = history + [f"User: {state['question']}", f"Relevant tables: {relevant_subset}"]
    return {
        "question": state["question"],
        "query": state["query"],
        "result": state["result"],
        "answer": state["answer"],
        "history": ensure_str_list(new_history),
        "table_list": relevant_subset
    }

# Step 2: Generate SQL query using schema subset
def generate_query(state: State) -> State:
    """Generate SQL query to fetch information using schema subset."""
    history: list[str] = ensure_str_list(state.get("history", []))
    table_list = state.get("table_list", [])
    table_list_comments = [t for t in TABLE_INFO if t.get('table_name') in table_list]
    db_schema_str = format_schema_description(table_list_comments, COLUMNS_INFO)

    # Convert table_list list to string for prompt
    prompt = query_prompt_template.invoke(
        {
            "input": state["question"],
            "history": history,
            "db_schema_str": db_schema_str,
            "schema": REDSHIFT_SCHEMA,
            "schema_comments": SCHEMA_COMMENTS
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    new_history: list[str] = history + [f"User: {state['question']}", f"SQL: {result['query']}"]
    return {
        "question": state["question"],
        "query": result["query"],
        "result": state["result"],
        "answer": state["answer"],
        "history": ensure_str_list(new_history),
        "table_list": table_list
    }

# Function to execute the SQL query
def execute_query(state: State) -> State:
    """Execute the SQL query using Redshift Data API and return the result."""
    result = query_redshift(state["query"])
    result_str = str(result)
    history: list[str] = ensure_str_list(state.get("history", []))
    new_history: list[str] = history + [f"SQL: {state['query']}", f"Result: {result_str}"]
    return {
        "question": state["question"],
        "query": state["query"],
        "result": result_str,
        "answer": state["answer"],
        "history": ensure_str_list(new_history),
        "table_list": state.get("table_list", [])
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
    prompt = answer_prompt_template.invoke(
        {
            "question": state["question"],
            "result": state["result"],
            "history": history
        }
    )
    answer = llm.invoke(prompt).content
    new_history: list[str] = history + [f"Answer: {answer}"]
    return {
        "question": str(state["question"]),
        "query": str(state["query"]),
        "result": str(state["result"]),
        "answer": str(answer),
        "history": ensure_str_list(new_history),
        "table_list": state.get("table_list", [])
    }

# Build the LangGraph workflow
workflow = StateGraph(State)

workflow.add_node("select_tables", select_tables_vector)
workflow.add_node("generate_query", generate_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("select_tables")
workflow.add_edge("select_tables", "generate_query")
workflow.add_edge("generate_query", "execute_query")
workflow.add_edge("execute_query", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()


# Example usage
if __name__ == "__main__":
    # Interactive loop for user queries
    print("\n💬 Ask me questions about your database! (type 'exit' or 'quit' to stop)")
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
            "table_list": []
        }
        result = app.invoke(state)
        print(f"\n📊 Answer: {result['answer']}")
        # Update history for next turn, always as list[str]
        history = ensure_str_list(result["history"])