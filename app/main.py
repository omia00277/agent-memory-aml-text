import logging
from fastapi import FastAPI, HTTPException, status
from app.schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from app.memory_service import add_memory, search_memory
from app.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agent Memory Leaderboard - Text Track Memory System",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Health check endpoint (no auth)."""
    return {"status": "ok"}


@app.post("/add", response_model=AddResponse)
async def add(request: AddRequest):
    """Add a memory chunk. Synchronous: returns only after data is searchable."""
    try:
        return add_memory(request)
    except Exception as e:
        logger.exception("/add failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search relevant memories within a user scope."""
    try:
        return search_memory(request)
    except Exception as e:
        logger.exception("/search failed")
        raise HTTPException(status_code=500, detail=str(e))
