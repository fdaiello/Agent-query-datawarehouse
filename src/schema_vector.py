
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Utility to build a FAISS vector store from schema information

def create_vectorstore(table_info: list) -> FAISS:
    """
    Build a FAISS vector store from a list of dicts with table_name and table_comment.
    Each document is table_name + comment, with table_name as metadata.
    """
    docs = [
        Document(
            page_content=f"{t['table_name']}: {t.get('table_comment', '')}",
            metadata={"table_name": t['table_name']}
        )
        for t in table_info if t.get('table_name')
    ]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def search_vectorstore(query: str, vectorstore: FAISS, top_k: int = 5) -> list:
    """
    Use vector search to select relevant tables for the query.
    Returns a list of table names from metadata.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return [r.metadata.get("table_name") for r in results if r.metadata.get("table_name")]
