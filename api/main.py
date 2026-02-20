"""
MentorZero - Multi-Agent AI Research Assistant
Main FastAPI application
"""
import logging
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect # type: ignore
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)