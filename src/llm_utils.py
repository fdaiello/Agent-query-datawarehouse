import os

def get_llm():
    """
    Initialize and return the LLM object based on environment variables.
    Supports OpenAI and Bedrock providers.
    """
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4.1")
        return ChatOpenAI(model=OPENAI_MODEL_ID, temperature=0)
    elif LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrock
        BEDROCK_PROVIDER = os.getenv("BEDROCK_PROVIDER", "anthropic")
        BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
        BEDROCK_INFERENCE_PROFILE_ID = os.getenv("BEDROCK_INFERENCE_PROFILE_ID")
        bedrock_kwargs = {
            "model_id": BEDROCK_INFERENCE_PROFILE_ID,
            "provider": BEDROCK_PROVIDER,
            "region": BEDROCK_REGION,
            "temperature": 0
        }
        return ChatBedrock(**bedrock_kwargs)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
