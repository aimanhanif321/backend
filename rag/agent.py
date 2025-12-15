import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    set_tracing_disabled,
    enable_verbose_stdout_logging,
)

enable_verbose_stdout_logging()
load_dotenv()
set_tracing_disabled(disabled=True)

# ---------- MODEL (Gemini via OpenAI-compatible API) ----------
provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=provider,
)

# ---------- COHERE ----------
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

def get_embedding(text: str):
    res = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[text],
    )
    return res.embeddings[0]

# ---------- QDRANT ----------
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=5,
)

@function_tool
def retrieve(query: str):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="humanoid_ai_book",
        query=embedding,
        limit=5,
    )
    return [p.payload.get("text", "") for p in result.points]

# ---------- AGENT (NO RUN CODE HERE) ----------
agent = Agent(
    name="Assistant",
    instructions="Summarize the provided context about Physical AI. Use ONLY the information given.",
    model=model,
    tools=[retrieve],
)


