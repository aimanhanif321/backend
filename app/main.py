import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure backend root is on path so rag package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from rag.agent import agent, Runner
except Exception as e:
    print("Failed to import rag.agent:")
    import traceback
    traceback.print_exc()
    agent = None
    Runner = None

app = FastAPI()

class ChatbotQueryRequest(BaseModel):
    query: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "API running"}

@app.post("/api/v1/chatbot/query")
async def chatbot_query(request: ChatbotQueryRequest):
    print("FRONTEND SENT:", request.query.strip())

    if not agent or not Runner:
        return {"error": "RAG agent not available on server."}

    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: Runner.run_sync(agent, input=request.query.strip())
            )

        final = getattr(result, "final_output", None) or getattr(result, "output", None)
        print("BACKEND RESPONSE:", final)

        return {"response": final or "No answer found."}

    except Exception as e:
        print("AGENT ERROR:", e)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
