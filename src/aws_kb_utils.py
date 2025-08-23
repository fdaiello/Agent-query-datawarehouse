"""
AWS Knowledge Base utilities for RAG functionality.
"""
import boto3
from typing import Dict, List, Any, Optional
from botocore.exceptions import ClientError
import os

# AWS Knowledge Base configuration
AWS_KNOWLEDGE_BASE_ID = os.getenv("AWS_KNOWLEDGE_BASE_ID", "")
AWS_REGION = os.getenv("AWS_REGION", "")

def get_bedrock_agent_runtime_client():
    """Initialize and return the Bedrock Agent Runtime client."""
    try:
        client = boto3.client(
            "bedrock-agent-runtime",
            region_name=AWS_REGION
        )
        return client
    except Exception as e:
        print(f"Error initializing Bedrock Agent Runtime client: {e}")
        raise

def query_bedrock_knowledge_base(question: str, knowledge_base_id: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Query the AWS Knowledge Base and return the results.
    
    Args:
        question (str): The user's question
        knowledge_base_id (str, optional): Knowledge Base ID. Defaults to environment variable.
        max_results (int): Maximum number of results to return
        
    Returns:
        Dict containing the query results and retrieved context
    """
    if not knowledge_base_id:
        knowledge_base_id = AWS_KNOWLEDGE_BASE_ID
    
    if not knowledge_base_id:
        raise ValueError("AWS_KNOWLEDGE_BASE_ID environment variable must be set")
    
    client = get_bedrock_agent_runtime_client()
    
    try:
        response = client.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={
                'text': question
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )
        
        # Extract the retrieved results
        retrieved_results = response.get('retrievalResults', [])
        
        # Format the context from retrieved results
        context_chunks = []
        for result in retrieved_results:
            content = result.get('content', {}).get('text', '')
            score = result.get('score', 0)
            metadata = result.get('metadata', {})
            
            context_chunks.append({
                'content': content,
                'score': score,
                'metadata': metadata
            })
        
        return {
            'question': question,
            'retrieved_results': context_chunks,
            'context': '\n\n'.join([chunk['content'] for chunk in context_chunks if chunk['content']])
        }
        
    except ClientError as e:
        print(f"AWS Client Error: {e}")
        raise
    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        raise

def retrieve_and_generate(question: str) -> Dict[str, Any]:
    """
    Use the retrieve and generate functionality of AWS Knowledge Base.
    
    Args:
        question (str): The user's question
        knowledge_base_id (str, optional): Knowledge Base ID
        model_arn (str, optional): Model ARN for generation
        
    Returns:
        Dict containing the generated answer and citations
    """
    knowledge_base_id = AWS_KNOWLEDGE_BASE_ID
    if not knowledge_base_id:
        raise ValueError("AWS_KNOWLEDGE_BASE_ID environment variable must be set")
    
    model_arn = os.getenv("BEDROCK_INFERENCE_PROFILE_ID")
    if not model_arn:
        raise ValueError("BEDROCK_INFERENCE_PROFILE_ID environment variable must be set")
    
    client = get_bedrock_agent_runtime_client()
    
    try:
        response = client.retrieve_and_generate(
            input={
                'text': question
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': model_arn
                }
            }
        )
        
        # Extract the generated output
        output = response.get('output', {})
        generated_text = output.get('text', '')
        
        # Extract citations
        citations = response.get('citations', [])
        
        return {
            'question': question,
            'answer': generated_text,
            'citations': citations,
            'session_id': response.get('sessionId', '')
        }
        
    except ClientError as e:
        print(f"AWS Client Error: {e}")
        raise
    except Exception as e:
        print(f"Error in retrieve and generate: {e}")
        raise

def format_citations(citations: List[Dict]) -> str:
    """
    Format citations for display.
    
    Args:
        citations (List[Dict]): List of citation objects
        
    Returns:
        str: Formatted citations string
    """
    if not citations:
        return ""
    
    formatted_citations = []
    for i, citation in enumerate(citations, 1):
        retrieved_references = citation.get('retrievedReferences', [])
        for ref in retrieved_references:
            content = ref.get('content', {}).get('text', '')[:200] + "..."
            metadata = ref.get('metadata', {})
            source = metadata.get('source', 'Unknown source')
            
            formatted_citations.append(f"[{i}] {source}: {content}")
    
    return "\n".join(formatted_citations)



