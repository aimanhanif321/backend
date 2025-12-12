# app/main.py
import os
import re
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# keep rag import — ensure rag/agent.py exists in repo
try:
    from rag.agent import agent, Runner
except Exception:
    agent = None
    Runner = None

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

class CitationRequest(BaseModel):
    text_data: str

class ChatbotQueryRequest(BaseModel):
    query: str

app = FastAPI(title="RAG FastAPI Backend")

# CORS origins come from env (comma separated)
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001")
CORS_ORIGINS = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_apa_citation(source_text: str) -> str:
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
    # Redis (for fastapi-cache). Use REDIS_URL env var (required in prod)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        redis = aioredis.from_url(redis_url, decode_responses=False)
        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.exception("Could not initialize Redis backend for fastapi-cache. Cache disabled. Error: %s", e)

    # Qdrant url and API key available via env (example usage by rag agent)
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        logger.info("Qdrant configured.")
    else:
        logger.warning("No QDRANT_URL found — vector DB functionality may not work in prod.")

@app.get("/")
def read_root():
    return {"status": "API running"}

@app.post("/api/v1/cite")
@cache(expire=60)
def create_citation(request: CitationRequest):
    citation = generate_apa_citation(request.text_data)
    return {"citation": citation}

@app.post("/api/v1/chatbot/query")
async def chatbot_query(request: ChatbotQueryRequest):
    if Runner is None or agent is None:
        return {"error": "RAG agent not available on server. Check rag/agent.py import."}

    # Use Runner.run if it's async; fallback to run_sync if necessary.
    try:
        # prefer async run if available
        if hasattr(Runner, "run") and callable(Runner.run):
            result = await Runner.run(agent, input=request.query)
        else:
            # fallback to sync runner executed in threadpool
            from concurrent.futures import ThreadPoolExecutor
            loop = __import__("asyncio").get_event_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, lambda: Runner.run_sync(agent, input=request.query))
    except Exception as e:
        logger.exception("Error running RAG agent: %s", e)
        return {"error": "Agent failure", "details": str(e)}

    # result may vary depending on Runner implementation
    final = getattr(result, "final_output", None) or getattr(result, "output", None) or str(result)
    return {"response": final}

# run with uvicorn in local/dev; Vercel will use different runner (see README)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
