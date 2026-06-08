import asyncio
import time
import subprocess
import logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("wake_proxy")

# Configuration
PROXY_PORT = 8080
BACKEND_URL = "http://localhost:8081"
TIMEOUT_SECONDS = 30 * 60  # 30 minutes
DOCKER_CWD = r"c:\Users\cocko\workspace\graphrag-aero"

app = FastAPI(title="Wake Proxy")
client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=120.0)

# State
last_activity_time = time.time()
# Assume awake at startup to avoid immediately issuing start commands if already running
is_awake = True

async def check_backend_healthy() -> bool:
    """Check if the backend is reachable and healthy."""
    try:
        # We query the health endpoint of the backend
        resp = await client.get("/healthz")
        return resp.status_code == 200 and resp.json().get("ok") is True
    except Exception:
        return False

async def ensure_awake():
    """Ensure the Docker containers are running before forwarding."""
    global is_awake
    if not is_awake:
        logger.info("Containers are asleep. Waking them up...")
        # Start the backend explicitly, which respects depends_on for DBs
        subprocess.run(
            ["docker", "compose", "up", "-d", "backend"], 
            cwd=DOCKER_CWD, 
            check=False
        )
        
        # Wait for the backend to be healthy
        logger.info("Waiting for backend to become healthy...")
        for i in range(60):
            if await check_backend_healthy():
                logger.info("Backend is fully awake and ready!")
                is_awake = True
                return
            await asyncio.sleep(1)
        
        logger.warning("Backend did not become healthy within 60 seconds. Proceeding anyway...")
        is_awake = True

async def sleep_monitor():
    """Background task to stop containers after inactivity."""
    global is_awake
    while True:
        await asyncio.sleep(60)
        
        if is_awake and (time.time() - last_activity_time) > TIMEOUT_SECONDS:
            logger.info(f"Idle timeout reached ({TIMEOUT_SECONDS}s). Stopping Docker containers...")
            # We issue a general stop to pause the environment and free RAM
            subprocess.run(
                ["docker", "compose", "stop"], 
                cwd=DOCKER_CWD, 
                check=False
            )
            is_awake = False
            logger.info("Containers are now asleep.")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(sleep_monitor())

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(request: Request, path: str):
    """Proxy all requests to the backend, waking it up if necessary."""
    global last_activity_time
    last_activity_time = time.time()
    
    await ensure_awake()

    # Construct the URL to forward to
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    
    # Build the request
    req = client.build_request(
        request.method,
        url,
        headers=request.headers.raw,
        content=request.stream()
    )
    
    # Send and stream the response back
    # This handles both normal JSON responses and Server-Sent Events (Streaming)
    resp = await client.send(req, stream=True)
    
    # Filter hop-by-hop and encoding headers that Starlette handles or conflicts with
    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    headers = {name: value for name, value in resp.headers.items() if name.lower() not in excluded_headers}
    
    return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        headers=headers,
        background=BackgroundTask(resp.aclose)
    )

if __name__ == "__main__":
    logger.info(f"Starting Wake Proxy on port {PROXY_PORT}")
    logger.info(f"Forwarding to backend on {BACKEND_URL}")
    uvicorn.run("wake_proxy:app", host="0.0.0.0", port=PROXY_PORT, reload=False)
