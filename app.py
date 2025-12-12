from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
from fastapi.middleware.cors import CORSMiddleware
import re
import sys
import os

# Add the rag directory to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.agent import agent, Runner # Import the agent and Runner from rag/agent.py

class CitationRequest(BaseModel):
    text_data: str

class ChatbotQueryRequest(BaseModel):
    query: str

app = FastAPI()

origins = ["http://localhost:3000", "http://localhost:3001"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_apa_citation(source_text):
    """
    Generates a basic APA-style citation from a given source text.
    This is a placeholder and would require a more sophisticated NLP/parsing approach
    for real-world citation generation.
    """
    # Example: Simple heuristic to extract author and year
    match = re.search(r'(?P<author>[A-Za-z\s.]+)\s+\((?P<year>\d{4})\)', source_text)
    if match:
        author = match.group('author').strip()
        year = match.group('year')
        title_match = re.search(r'\"(?P<title>[^\"]+)\"|\'(?P<title>[^\']+)\'', source_text)
        title = title_match.group('title') if title_match else "Untitled"
        return f"{author} ({year}). {title}. [Retrieved from context]."
    return f"[APA citation for: {source_text}]"

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

@app.get("/")
def read_root():
    return {"status": "API running"}

@app.post("/api/v1/cite")
@cache(expire=60)
def create_citation(request: CitationRequest):
    citation = generate_apa_citation(request.text_data)
    return {"citation": citation}

# @app.post("/api/v1/chatbot/query")
# async def chatbot_query(request: ChatbotQueryRequest):
#     result = Runner.run_sync(
#         agent,
#         input=request.query,
#     )
#     return {"response": result.final_output}

@app.post("/api/v1/chatbot/query")
async def chatbot_query(request: ChatbotQueryRequest):
     # Runner.run_sync ko Runner.run mein badlen aur aage 'await' laga dein
     result = await Runner.run( 
         agent,
         input=request.query,
     )
     return {"response": result.final_output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
