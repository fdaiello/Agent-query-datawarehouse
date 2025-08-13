from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Utility to build a FAISS vector store from schema_info

def build_schema_vectorstore(schema_info: str) -> FAISS:
    """
    Build a FAISS vector store from schema_info string.
    Each table/column line is treated as a document.
    """
    # Split schema_info into lines for indexing
    docs = [line for line in schema_info.split('\n') if line.strip()]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore


def select_relevant_tables_vector(query: str, vectorstore: FAISS, top_k: int = 5) -> str:
    """
    Use vector search to select relevant tables/columns for the query.
    Returns a subset of schema_info as a string.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return '\n'.join([r.page_content for r in results])
