import logging
from fastapi import Depends, FastAPI
from api.startup import register_startup
from api.routes import router as api_router
from api.deps import get_ollama_client
from agent.llm.ollama_client import OllamaClient
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


app = FastAPI(title="MentorZero API", version="0.1.0")
register_startup(app)
app.include_router(api_router)
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/")
def read_root() -> dict:
	return {"service": "MentorZero API", "status": "ok"}


@app.get("/llm_health")
def llm_health(client: OllamaClient = Depends(get_ollama_client)) -> dict:
	return client.health()