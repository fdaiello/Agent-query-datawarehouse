
from typing import TypedDict, Annotated, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
load_dotenv()
from db_utils import get_redshift_schema_info, query_redshift, REDSHIFT_SCHEMA

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
Only use the following tables:{table_info}"""
user_prompt = "Question: {input}"
query_prompt_template = ChatPromptTemplate(
    [
        ("system", system_message),
        MessagesPlaceholder("history"),
        ("user", user_prompt)
    ]
)


class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]




# Function to write the SQL query
def write_query(state: State) -> State:
    """Generate SQL query to fetch information."""
    history: list[str] = ensure_str_list(state.get("history", []))
    schema_info = get_redshift_schema_info(REDSHIFT_SCHEMA)
    prompt = query_prompt_template.invoke(
        {
            "input": state["question"],
            "history": history,
            "table_info": schema_info,
            "schema": REDSHIFT_SCHEMA
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
        "history": ensure_str_list(new_history)
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
        "history": ensure_str_list(new_history)
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
        "history": ensure_str_list(new_history)
    }


# Build the LangGraph workflow
workflow = StateGraph(State)

workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("write_query")
workflow.add_edge("write_query", "execute_query")
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
            "history": safe_history
        }
        result = app.invoke(state)
        print(f"\n📊 Answer: {result['answer']}")
        # Update history for next turn, always as list[str]
        history = ensure_str_list(result["history"])