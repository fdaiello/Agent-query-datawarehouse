from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Utility to build a FAISS vector store from schema information

def create_vectorstore(text_data: str) -> FAISS:
    """
    Build a FAISS vector store from schema_info string.
    Each table/column line is treated as a document.
    """
    # Split schema_info into lines for indexing
    docs = [line for line in text_data.split('\n') if line.strip()]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)
    return vectorstore


def search_vectorstore(query: str, vectorstore: FAISS, top_k: int = 5) -> str:
    """
    Use vector search to select relevant tables/columns for the query.
    Returns a subset of schema_info as a string.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return '\n'.join([r.page_content for r in results])
