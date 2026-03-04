"""
MentorZero - Multi-Agent AI Research Assistant
Main FastAPI application
"""
import logging
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Request # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import asyncio
from api.agent_routes import router as agent_router # type: ignore
import uvicorn # type: ignore
from api.agent_routes import initialize_agents # Re-added initialize_agents import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MentorZero",
    description="Multi-Agent AI Research Assistant",
    version="2.0.0"
)

@app.websocket("/ws_test")
async def websocket_test(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Hello from test")
    await websocket.close()

# Configure CORS: Restrict to localhost by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
)

# Custom Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Basic CSP: Allow self and font/icon CDNs
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' ws://localhost:8000 ws://127.0.0.1:8000 http://localhost:8000 http://127.0.0.1:8000; "
        "img-src 'self' data: https:;"
    )
    return response

# Mount static files for UI
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# Include API routes
app.include_router(agent_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "MentorZero",
        "version": "2.0.0"
    }

# Root redirect to UI
@app.get("/")
async def root():
    return {
        "message": "Welcome to MentorZero",
        "docs": "/docs",
        "ui": "/ui/modern.html"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting MentorZero Multi-Agent System...")
    await initialize_agents()
    logger.info("All agents initialized and ready!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down MentorZero...")

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000) # type: ignore