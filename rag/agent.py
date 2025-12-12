from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
from agents import enable_verbose_stdout_logging
import time
import cohere
from qdrant_client import QdrantClient
enable_verbose_stdout_logging()

load_dotenv()
set_tracing_disabled(disabled=True)

print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:5]}...") # Print first 5 chars for security
print(f"COHERE_API_KEY: {os.getenv('COHERE_API_KEY')[:5]}...") # Print first 5 chars for security
print(f"QDRANT_URL: {os.getenv('QDRANT_URL')}")
print(f"QDRANT_API_KEY: {os.getenv('QDRANT_API_KEY')[:5]}...") # Print first 5 chars for security

gemini_api_key = os.getenv("OPENAI_API_KEY")
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=provider
)




# Initialize Cohere client
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    try:
        response = cohere_client.embed(
            model="embed-english-v3.0",
            input_type="search_query",  # Use search_query for queries
            texts=[text],
        )
        print(f"Cohere embedding successful for text: {text[:30]}...")
        return response.embeddings[0]  # Return the first embedding
    except Exception as e:
        print(f"Error during Cohere embedding for text: {text[:30]}... Error: {e}")
        raise  # Re-raise the exception to propagate it


@function_tool
def retrieve(query):
    try:
        embedding = get_embedding(query)
        result = qdrant.query_points(
            collection_name="humanoid_ai_book",
            query=embedding,
            limit=5
        )
        print("Retrieved points:", result.points)
        return [point.payload["text"] for point in result.points]
    except Exception as e:
        print(f"Error during Qdrant retrieval for query: {query}. Error: {e}")
        raise # Re-raise the exception to propagate it




agent = Agent(
    name="Assistant",
    instructions="Summarize the provided context about Physical AI. Use ONLY the information given.",
    model=model,
    tools=[retrieve]
)




