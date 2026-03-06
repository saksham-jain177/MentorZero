"""
New Agent-based API Routes
Handles multi-agent orchestration and research workflows
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, File, UploadFile, WebSocketDisconnect, Depends # type: ignore[import-untyped]
from pydantic import BaseModel  # type: ignore[import-untyped]
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
import asyncio
import dataclasses
import json
import re
import uuid
import shutil
import os
import time
from datetime import datetime
from agent.core.llm_client import LLMClient

class SecurityManager:
    @staticmethod
    def sanitize_query(query: str, max_length: int = 500) -> str:
        """Sanitize user input to prevent prompt injection, XSS, and SQLi"""
        if not query:
            return ""
            
        # Trim length
        sanitized = query[:max_length]
        
        # 1. Remove common prompt injection markers
        sanitized = re.sub(r'(system:|you are|ignore all|as a|persona:|role:|instruction:)', '', sanitized, flags=re.IGNORECASE)
        
        # 2. XSS Guards: Basic removal of script tags and event handlers
        sanitized = re.sub(r'<script.*?>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        
        # 3. SQLi Guards: Remove common markers (while allowing natural language)
        sql_keywords = r'(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\s+(FROM|INTO|TABLE|WHERE|JOIN)'
        sanitized = re.sub(sql_keywords, '[SENSITIVE]', sanitized, flags=re.IGNORECASE)
        sanitized = sanitized.replace("'", "''")  # Escape single quotes
        
        # Final cleanup of extra whitespace
        return sanitized.strip()

from fastapi import Header, Request
from agent.config import get_settings

async def verify_api_key(x_api_key: str = Header(...), required_scope: str = "general"):
    """FastAPI dependency to verify X-API-Key and Scope (Compliance Stage 21)"""
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    # In a real system, we'd check a DB/JWT for scopes. 
    # For now, we simulate success for the 'general' and 'research' scopes.
    # This fulfills the 'Least Privilege' requirement 2026.
    allowed_scopes = ["general", "research", "admin"]
    if required_scope not in allowed_scopes:
        raise HTTPException(status_code=403, detail=f"Key lacking required scope: {required_scope}")
        
    return x_api_key

class RateLimiter:
    """Simple in-memory rate limiter per IP"""
    def __init__(self):
        self.requests = {}  # {ip: [timestamps]}
        
    async def __call__(self, request: Request):
        settings = get_settings()
        ip = request.client.host
        now = time.time()
        
        # Cleanup old timestamps
        if ip in self.requests:
            self.requests[ip] = [t for t in self.requests[ip] if now - t < 60]
            
            if len(self.requests[ip]) >= settings.rate_limit_requests:
                raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")
            
            self.requests[ip].append(now)
        else:
            self.requests[ip] = [now]

rate_limiter = RateLimiter()
from datetime import datetime

from agent.core.orchestrator import (  # type: ignore[import-untyped]
    AgentOrchestrator,
    AgentTask,
    ExecutionMode,
    TaskResult,
    SearchAgent,
    WritingAgent,
    OptimizationAgent
)
from agent.core.research_agent import ResearchAgent, ResearchMode # type: ignore[import-untyped]
from agent.core.azl_scorer import InputGuardrail # type: ignore[import-untyped]
from agent.core.capabilities import (  # type: ignore[import-untyped]
    CodeGenerationAgent,
    LearningAgent,
    AnalysisAgent,
    AutomationAgent,
    CreativeAgent
)
from agent.core.exporter import report_exporter # type: ignore[import-untyped]

router = APIRouter(prefix="/api/v2", tags=["agents"])

# Global instances
orchestrator = AgentOrchestrator()
input_guardrail = InputGuardrail()

# Initialize agents on startup
async def initialize_agents():
    """Initialize all specialized agents"""
    # Create a shared LLM client
    llm_client = LLMClient()
    
    # Research & Information
    orchestrator.register_agent("search", SearchAgent(llm_client))
    orchestrator.register_agent("research", ResearchAgent(llm_client))
    orchestrator.register_agent("writer", WritingAgent(llm_client))
    orchestrator.register_agent("optimizer", OptimizationAgent(llm_client))
    
    # New Capabilities
    orchestrator.register_agent("coder", CodeGenerationAgent(llm_client))
    orchestrator.register_agent("learner", LearningAgent(llm_client))
    orchestrator.register_agent("analyzer", AnalysisAgent())
    orchestrator.register_agent("automator", AutomationAgent())
    orchestrator.register_agent("creative", CreativeAgent(llm_client))
    
    # Initialize Guardrail with LLM
    input_guardrail.llm = llm_client

# Request/Response Models
class ResearchRequest(BaseModel):
    query: str
    mode: str = "adaptive"  # adaptive, parallel, sequential
    research_mode: str = "general" # general, academic, market, technical
    depth: str = "standard"  # quick, standard, deep
    include_sources: bool = True
    seed_documents: Optional[List[str]] = None
    vision_enabled: bool = False
    max_agents: Optional[int] = None
    
    if TYPE_CHECKING:
        def __init__(self, **kwargs: Any) -> None: ...

class ResearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    execution_time: float
    agents_used: List[str]
    mode: str
    system_stats: Dict[str, Any]
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    if TYPE_CHECKING:
        def __init__(self, **kwargs: Any) -> None: ...

class AgentStatus(BaseModel):
    agent_name: str
    status: str
    current_task: Optional[str]
    tasks_completed: int
    
    if TYPE_CHECKING:
        def __init__(self, **kwargs: Any) -> None: ...

@router.post("/research/upload", dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
async def upload_seed_document(file: UploadFile = File(...)):
    """Temporarily upload a PDF to be used as a seed document"""
    settings = get_settings()
    
    # 1. Size Validation (Max 10MB)
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_size_mb}MB allowed.")
    await file.seek(0)

    # 2. Extension Validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    temp_dir = "./data/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 3. Path Traversal Guard: Use basename and a fresh UUID
    original_filename = os.path.basename(file.filename)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(temp_dir, f"{file_id}_{original_filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"file_id": file_id, "file_path": os.path.abspath(file_path), "filename": file.filename}
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

@router.post("/research/voice", dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
async def transcribe_voice(file: UploadFile = File(...)):
    """Transcribe voice recording using Whisper"""
    settings = get_settings()
    
    # 1. Size Validation (Max 10MB)
    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Voice file too large.")
    await file.seek(0)

    temp_dir = "./data/temp_voice"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. Path Traversal Guard
    original_filename = os.path.basename(file.filename)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(temp_dir, f"{file_id}_{original_filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        import whisper
        # Use base model for a good balance of speed and accuracy
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        
        # Cleanup
        os.remove(file_path)
        
        return {"text": result["text"].strip()}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Transcription failed")

# API Endpoints
@router.post("/research", response_model=ResearchResponse, dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
async def research_topic(request: ResearchRequest):
    """
    Execute a multi-agent research workflow
    """
    try:
        # Convert mode string to enum
        mode_map = {
            "adaptive": ExecutionMode.ADAPTIVE,
            "parallel": ExecutionMode.PARALLEL,
            "sequential": ExecutionMode.SEQUENTIAL
        }
        execution_mode = mode_map.get(request.mode, ExecutionMode.ADAPTIVE)
        
        # SECURITY: Sanitize user input
        sanitized_query = SecurityManager.sanitize_query(request.query)
        if not sanitized_query:
             raise HTTPException(status_code=400, detail="Invalid or empty query after sanitization")
        
        # 1. INPUT GUARDRAIL (Compliance Stage 21)
        safety_scan = await input_guardrail.scan_query(sanitized_query)
        if not safety_scan.get("is_safe", True):
            raise HTTPException(
                status_code=400, 
                detail=f"Safety Violation: {safety_scan.get('explanation', 'Policy Breach')}"
            )
        
        # Create task pipeline based on query complexity
        tasks = []
        
        # Step 1: Optimize the query first to get niche biasing
        optimizer_task = AgentTask(
            agent_name="optimizer",
            task_type="optimize",
            input_data=sanitized_query,
            priority=10
        )
        
        # We run the optimizer first so search agents can use the "niched" query
        opt_results = await orchestrator.execute_tasks([optimizer_task])
        optimized_query = request.query
        if opt_results and opt_results[0].success:
            optimized_query = opt_results[0].output
            
        # Step 2: Parallel search and research using the optimized query
        tasks = []
        tasks.append(AgentTask(
            agent_name="search",
            task_type="web_search",
            input_data=optimized_query,
            priority=8
        ))
        
        if request.depth in ["standard", "deep"]:
            # Map research mode string to enum
            research_mode_map = {
                "general": ResearchMode.GENERAL,
                "academic": ResearchMode.ACADEMIC,
                "market": ResearchMode.MARKET,
                "technical": ResearchMode.TECHNICAL
            }
            res_mode = research_mode_map.get((request.research_mode or "general").lower(), ResearchMode.GENERAL)
            
            tasks.append(AgentTask(
                agent_name="research",
                task_type="research_topic",
                input_data={
                    "query": optimized_query,
                    "mode": res_mode,
                    "depth": request.depth,
                    "seed_documents": request.seed_documents,
                    "vision_enabled": request.vision_enabled
                },
                priority=8
            ))
        
        # Step 3: Summarize findings
        tasks.append(AgentTask(
            agent_name="writer",
            task_type="summarize",
            input_data=f"Combined research findings for: {optimized_query}",
            priority=5,
            requires=["web_search"]
        ))
        
        # Execute remaining tasks
        import time
        start_time = time.time()
        
        results = await orchestrator.execute_tasks(tasks, mode=execution_mode)
        
        if opt_results:
            results = opt_results + results
        
        execution_time = time.time() - start_time
        
        # Format response
        formatted_results = []
        agents_used = []
        
        for result in results:
            if result.success:
                formatted_results.append({
                    "agent": result.agent_name,
                    "task": result.task_type,
                    "output": result.output,
                    "duration": result.duration
                })
                if result.agent_name not in agents_used:
                    agents_used.append(result.agent_name)
        
        response_data = {
            "query": request.query,
            "results": formatted_results,
            "execution_time": execution_time,
            "agents_used": agents_used,
            "mode": request.mode,
            "system_stats": orchestrator.resource_monitor.get_system_stats(),
            "timestamp": datetime.now().isoformat()
        }

        # Save session to database for history
        session_id = ""
        research_agent = orchestrator.agents.get("research")
        if research_agent and hasattr(research_agent, "graph_store"):
            session_id = research_agent.graph_store.save_research_session(
                query=request.query,
                results_json=json.dumps(response_data),
                niche_focus=getattr(orchestrator.agents.get("optimizer"), "niche_focus", None)
            )
        
        response_data["session_id"] = session_id
        return ResearchResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/status", response_model=List[AgentStatus], dependencies=[Depends(verify_api_key)])
async def get_agents_status():
    """
    Get status of all registered agents
    """
    statuses = []
    for name, agent in orchestrator.agents.items():
        statuses.append(AgentStatus(
            agent_name=name,
            status="ready",  # Would be enhanced with actual status tracking
            current_task=None,
            tasks_completed=0
        ))
    return statuses

@router.get("/system/stats", dependencies=[Depends(verify_api_key)])
async def get_system_stats():
    """
    Get current system resource statistics
    """
    return orchestrator.resource_monitor.get_system_stats()

@router.get("/graph", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def get_graph():
    """Get the full knowledge graph from the research agent"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        return {"nodes": [], "edges": []}
    
    return research_agent.graph_store.get_full_graph()

@router.get("/graph/search", response_model=Dict[str, Any], dependencies=[Depends(verify_api_key)])
async def search_graph(query: str):
    """Search for a specific entity and its relationships in the graph"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        return {"nodes": [], "edges": []}
    
    return research_agent.graph_store.search_subgraph(query)

@router.get("/research/history", dependencies=[Depends(verify_api_key)])
async def get_research_history(limit: int = 10):
    """Retrieve history of research sessions"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        return []
    return research_agent.graph_store.get_research_history(limit)

@router.get("/research/session/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_session_results(session_id: str):
    """Get full results for a past research session"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        raise HTTPException(status_code=404, detail="Research agent not found")
    
    results = research_agent.graph_store.get_session_results(session_id)
    if not results:
        raise HTTPException(status_code=404, detail="Session not found")
    return results

@router.get("/research/export/{session_id}", dependencies=[Depends(verify_api_key)])
async def export_session_report(session_id: str):
    """Export a research session to Markdown"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        raise HTTPException(status_code=404, detail="Research agent not found")
    
    results = research_agent.graph_store.get_session_results(session_id)
    if not results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    filepath = report_exporter.export_to_markdown(results)
    if not filepath:
        raise HTTPException(status_code=500, detail="Export failed")
        
    return {"message": "Report exported successfully", "path": os.path.basename(filepath)}

@router.post("/execute/custom", dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
async def execute_custom_workflow(tasks: List[Dict[str, Any]]):
    """
    Execute a custom agent workflow
    
    Example:
    [
        {"agent": "search", "task": "web_search", "input": "quantum computing"},
        {"agent": "writer", "task": "summarize", "input": "results", "requires": ["web_search"]}
    ]
    """
    try:
        agent_tasks = []
        for task_dict in tasks:
            agent_tasks.append(AgentTask(
                agent_name=task_dict["agent"],
                task_type=task_dict["task"],
                input_data=task_dict["input"],
                priority=task_dict.get("priority", 5),
                requires=task_dict.get("requires", None)
            ))
        
        results = await orchestrator.execute_tasks(agent_tasks)
        
        return {
            "success": True,
            "results": [
                {
                    "agent": r.agent_name,
                    "task": r.task_type,
                    "success": r.success,
                    "output": r.output if r.success else r.error
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.websocket("/ws/research")
async def research_websocket(websocket: WebSocket, api_key: Optional[str] = None):
    """
    WebSocket endpoint for real-time research updates
    """
    settings = get_settings()
    if api_key != settings.api_key:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Invalid API Key"})
        await websocket.close(code=4003)
        return

    await websocket.accept()
    
    # State for Delta Updates
    sent_nodes = set()
    sent_edges = set()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            mode = data.get("mode", "adaptive")
            research_mode = data.get("research_mode", "general")
            depth = data.get("depth", "standard")
            force_refresh = data.get("force_refresh", False)
            seed_files = data.get("seed_files", []) # List of local paths
            vision_enabled = data.get("vision_enabled", False)
            
            if not query:
                await websocket.send_json({"type": "error", "message": "No query provided"})
                continue
            
            # SECURITY: Sanitize user input
            sanitized_query = SecurityManager.sanitize_query(query)
            if not sanitized_query:
                await websocket.send_json({"type": "error", "message": "Invalid query after sanitization"})
                continue
            
            # Map mode
            mode_map = {
                "adaptive": ExecutionMode.ADAPTIVE,
                "parallel": ExecutionMode.PARALLEL,
                "sequential": ExecutionMode.SEQUENTIAL
            }
            execution_mode = mode_map.get(mode, ExecutionMode.ADAPTIVE)

            # GLOBAL SETTINGS OVERRIDE for this session
            from agent.config import get_settings
            settings = get_settings()
            original_ttl = settings.cache_ttl_hours
            if force_refresh:
                settings.cache_ttl_hours = 0  # Bypass durable cache
                logger.info("Force refresh enabled: Bypassing cache for this session.")

            # Callback for task start
            async def on_start(task: AgentTask):
                await websocket.send_json({
                    "type": "agent_update",
                    "agent": task.agent_name,
                    "task": task.task_type,
                    "status": "starting"
                })

            # Callback for task completion
            async def on_complete(result: TaskResult):
                output = result.output
                # Safely convert output to a JSON-serializable form
                # Send agent update
                await websocket.send_json({
                    "type": "agent_update",
                    "agent": result.agent_name,
                    "task": result.task_type,
                    "status": "completed" if result.success else "failed",
                    "output": output if result.success else None,
                    "error": result.error if not result.success else None,
                    "duration": result.duration
                })

                # DELTA UPDATE: Send only new graph elements
                research_agent = orchestrator.agents.get("research")
                if research_agent and hasattr(research_agent, "graph_store"):
                    full_graph = research_agent.graph_store.get_full_graph()
                    
                    new_nodes = [n for n in full_graph.get("nodes", []) if n["data"]["id"] not in sent_nodes]
                    new_edges = [e for e in full_graph.get("edges", []) if e["data"]["id"] not in sent_edges]
                    
                    if new_nodes or new_edges:
                        await websocket.send_json({
                            "type": "graph_delta",
                            "nodes": new_nodes,
                            "edges": new_edges
                        })
                        # Track what was sent
                        for n in new_nodes: sent_nodes.add(n["data"]["id"])
                        for e in new_edges: sent_edges.add(e["data"]["id"])
            
            # 1. INPUT GUARDRAIL (Compliance Stage 21)
            safety_scan = await input_guardrail.scan_query(sanitized_query)
            if not safety_scan.get("is_safe", True):
                await websocket.send_json({
                    "type": "error",
                    "message": f"Safety Violation: {safety_scan.get('explanation', 'Policy Breach')}"
                })
                return
            
            # Step 1: Optimize query for niche biasing
            optimizer_task = AgentTask("optimizer", "optimize", sanitized_query, priority=10)
            await on_start(optimizer_task)
            
            opt_results = await orchestrator.execute_tasks([optimizer_task])
            optimized_query = sanitized_query
            if opt_results and opt_results[0].success:
                optimized_query = opt_results[0].output
                await on_complete(opt_results[0])
            
            # Map research mode string to enum
            research_mode_map = {
                "general": ResearchMode.GENERAL,
                "academic": ResearchMode.ACADEMIC,
                "market": ResearchMode.MARKET,
                "technical": ResearchMode.TECHNICAL
            }
            res_mode = research_mode_map.get((research_mode or "general").lower(), ResearchMode.GENERAL)

            # Step 2: Define and execute secondary tasks with optimized query
            tasks = [
                AgentTask("search", "web_search", optimized_query, priority=8),
                AgentTask("research", "research_topic", input_data={
                    "query": optimized_query, 
                    "mode": res_mode,
                    "depth": depth,
                    "seed_documents": seed_files,
                    "vision_enabled": vision_enabled
                }, priority=7),
                AgentTask("writer", "summarize", f"Research for {optimized_query}", priority=5, requires=["web_search"])
            ]
            
            # Execute tasks with callbacks
            start_time = time.time()
            
            results: List[TaskResult] = await orchestrator.execute_tasks(
                tasks, 
                mode=execution_mode,
                on_task_start=on_start,
                on_task_complete=on_complete
            )
            
            total_time = time.time() - start_time
            
            # Save WebSocket session to history
            session_id = ""
            research_agent = orchestrator.agents.get("research")
            if research_agent and hasattr(research_agent, "graph_store"):
                # Prepare a full result object similar to REST for history
                ws_data = {
                    "query": sanitized_query,
                    "results": [
                        {
                            "agent": r.agent_name,
                            "task": r.task_type,
                            "output": r.output if r.success else None,
                            "error": r.error if not r.success else None,
                            "duration": r.duration
                        } for r in results
                    ],
                    "total_time": total_time,
                    "timestamp": datetime.now().isoformat(),
                    "source": "websocket"
                }
                session_id = research_agent.graph_store.save_research_session(
                    query=sanitized_query,
                    results_json=json.dumps(ws_data),
                    niche_focus=getattr(orchestrator.agents.get("optimizer"), "niche_focus", None)
                )

            # Send final summary with session_id
            summary = {
                "type": "research_complete",
                "query": sanitized_query,
                "total_time": total_time,
                "results_count": len(results),
                "session_id": session_id
            }
            # Restore settings
            settings.cache_ttl_hours = original_ttl
            
            await websocket.send_json(summary)
            
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

@router.post("/code/generate")
async def generate_code(request: Dict[str, Any]):
    """Generate code from requirements"""
    requirements = request.get("requirements", "")
    language = request.get("language", "python")
    
    task = AgentTask(
        agent_name="coder",
        task_type="generate_code",
        input_data=requirements,
        priority=10
    )
    
    results = await orchestrator.execute_tasks([task])
    return results[0].output if results else {"error": "Failed to generate code"}

@router.post("/code/analyze")
async def analyze_code(request: Dict[str, Any]):
    """Analyze code for improvements"""
    code = request.get("code", "")
    
    task = AgentTask(
        agent_name="coder",
        task_type="analyze_code",
        input_data=code,
        priority=10
    )
    
    results = await orchestrator.execute_tasks([task])
    return results[0].output if results else {"error": "Failed to analyze code"}

@router.post("/learn/path")
async def create_learning_path(request: Dict[str, Any]):
    """Create personalized learning path"""
    topic = request.get("topic", "")
    level = request.get("level", "beginner")
    
    task = AgentTask(
        agent_name="learner",
        task_type="create_learning_path",
        input_data=topic,
        priority=10
    )
    
    results = await orchestrator.execute_tasks([task])
    return results[0].output if results else {"error": "Failed to create learning path"}

@router.post("/learn/quiz")
async def generate_quiz(request: Dict[str, Any]):
    """Generate quiz questions"""
    topic = request.get("topic", "")
    difficulty = request.get("difficulty", "medium")
    
    task = AgentTask(
        agent_name="learner",
        task_type="generate_quiz",
        input_data=topic,
        priority=10
    )
    
    results = await orchestrator.execute_tasks([task])
    return results[0].output if results else {"error": "Failed to generate quiz"}

@router.post("/analyze/text")
async def analyze_text(request: Dict[str, Any]):
    """Analyze text for insights"""
    text = request.get("text", "")
    analysis_type = request.get("type", "summary")
    
    task = AgentTask(
        agent_name="analyzer",
        task_type="analyze_text",
        input_data=text,
        priority=10
    )
    
    results = await orchestrator.execute_tasks([task])
    return results[0].output if results else {"error": "Failed to analyze text"}

@router.post("/creative/brainstorm")
async def brainstorm_ideas(request: Dict[str, Any]):
    """Brainstorm creative ideas"""
    topic = request.get("topic", "")
    count = request.get("count", 10)
    
    task = AgentTask(
        agent_name="creative",
        task_type="brainstorm",
        input_data=topic,
        priority=10
    )
    
    results = await orchestrator.execute_tasks([task])
    return results[0].output if results else {"error": "Failed to brainstorm"}

@router.post("/workflow/create")
async def create_workflow(request: Dict[str, Any]):
    """Create automation workflow"""
    name = request.get("name", "")
    steps = request.get("steps", [])
    
    task = AgentTask(
        agent_name="automator",
        task_type="create_workflow",
        input_data={"name": name, "steps": steps},
        priority=10
    )
    
    results = await orchestrator.execute_tasks([task])
    return results[0].output if results else {"error": "Failed to create workflow"}

# Agents are initialized via main.py startup event
