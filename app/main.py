import logging
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
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

security = HTTPBearer(auto_error=False)


def _verify_auth(credentials: HTTPAuthorizationCredentials, request: Request):
    """Verify AML platform auth when memory_system_key is configured."""
    key = settings.memory_system_key
    if not key:
        return

    provided = None
    if credentials and credentials.scheme.lower() in ("bearer", "token"):
        provided = credentials.credentials
    elif request.headers.get("x-api-key"):
        provided = request.headers.get("x-api-key")
    elif request.headers.get("authorization"):
        auth = request.headers.get("authorization")
        provided = auth.split(None, 1)[-1].strip()

    if provided != key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )


async def auth_dependency(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    _verify_auth(credentials, request)


@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Health check endpoint (no auth)."""
    return {"status": "ok"}


@app.post("/add", response_model=AddResponse, dependencies=[Depends(auth_dependency)])
async def add(request: AddRequest):
    """Add a memory chunk. Synchronous: returns only after data is searchable."""
    try:
        return add_memory(request)
    except Exception as e:
        logger.exception("/add failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(auth_dependency)])
async def search(request: SearchRequest):
    """Search relevant memories within a user scope."""
    try:
        return search_memory(request)
    except Exception as e:
        logger.exception("/search failed")
        raise HTTPException(status_code=500, detail=str(e))
