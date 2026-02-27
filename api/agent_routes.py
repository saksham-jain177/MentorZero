"""
New Agent-based API Routes
Handles multi-agent orchestration and research workflows
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket  # type: ignore[import-untyped]
from pydantic import BaseModel  # type: ignore[import-untyped]
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
import asyncio
import dataclasses
import json
import time
import os
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
from agent.core.research_agent import ResearchAgent  # type: ignore[import-untyped]
from agent.core.capabilities import (  # type: ignore[import-untyped]
    CodeGenerationAgent,
    LearningAgent,
    AnalysisAgent,
    AutomationAgent,
    CreativeAgent
)
from agent.core.exporter import report_exporter # type: ignore[import-untyped]

router = APIRouter(prefix="/api/v2", tags=["agents"])

# Global orchestrator instance
orchestrator = AgentOrchestrator()

# Initialize agents on startup
async def initialize_agents():
    """Initialize all specialized agents"""
    # Research & Information
    orchestrator.register_agent("search", SearchAgent())
    orchestrator.register_agent("research", ResearchAgent())
    orchestrator.register_agent("writer", WritingAgent())
    orchestrator.register_agent("optimizer", OptimizationAgent())
    
    # New Capabilities
    orchestrator.register_agent("coder", CodeGenerationAgent())
    orchestrator.register_agent("learner", LearningAgent())
    orchestrator.register_agent("analyzer", AnalysisAgent())
    orchestrator.register_agent("automator", AutomationAgent())
    orchestrator.register_agent("creative", CreativeAgent())

# Request/Response Models
class ResearchRequest(BaseModel):
    query: str
    mode: str = "adaptive"  # adaptive, parallel, sequential
    depth: str = "standard"  # quick, standard, deep
    include_sources: bool = True
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

# API Endpoints
@router.post("/research", response_model=ResearchResponse)
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
        
        # Create task pipeline based on query complexity
        tasks = []
        
        # Step 1: Optimize the query first to get niche biasing
        optimizer_task = AgentTask(
            agent_name="optimizer",
            task_type="optimize_query",
            input_data=request.query,
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
            tasks.append(AgentTask(
                agent_name="research",
                task_type="research_topic",
                input_data=optimized_query,
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

@router.get("/agents/status", response_model=List[AgentStatus])
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

@router.get("/system/stats")
async def get_system_stats():
    """
    Get current system resource statistics
    """
    return orchestrator.resource_monitor.get_system_stats()

@router.get("/graph", response_model=Dict[str, Any])
async def get_graph():
    """Get the full knowledge graph from the research agent"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        return {"nodes": [], "edges": []}
    
    return research_agent.graph_store.get_full_graph()

@router.get("/graph/search", response_model=Dict[str, Any])
async def search_graph(query: str):
    """Search for a specific entity and its relationships in the graph"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        return {"nodes": [], "edges": []}
    
    return research_agent.graph_store.search_subgraph(query)

@router.get("/research/history")
async def get_research_history(limit: int = 10):
    """Retrieve history of research sessions"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        return []
    return research_agent.graph_store.get_research_history(limit)

@router.get("/research/session/{session_id}")
async def get_session_results(session_id: str):
    """Get full results for a past research session"""
    research_agent = orchestrator.agents.get("research")
    if not research_agent:
        raise HTTPException(status_code=404, detail="Research agent not found")
    
    results = research_agent.graph_store.get_session_results(session_id)
    if not results:
        raise HTTPException(status_code=404, detail="Session not found")
    return results

@router.get("/research/export/{session_id}")
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

@router.post("/execute/custom")
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
async def research_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time research updates
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            mode = data.get("mode", "adaptive")
            
            if not query:
                await websocket.send_json({"type": "error", "message": "No query provided"})
                continue
            
            # Map mode
            mode_map = {
                "adaptive": ExecutionMode.ADAPTIVE,
                "parallel": ExecutionMode.PARALLEL,
                "sequential": ExecutionMode.SEQUENTIAL
            }
            execution_mode = mode_map.get(mode, ExecutionMode.ADAPTIVE)

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
                if result.success and output is not None:
                    if dataclasses.is_dataclass(output):
                        # Convert dataclass to dict, serializing datetime fields as ISO strings
                        raw = dataclasses.asdict(output)
                        output = json.loads(json.dumps(raw, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o)))
                    elif hasattr(output, '__dict__'):
                        raw = output.__dict__
                        output = json.loads(json.dumps(raw, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o)))
                    elif not isinstance(output, (str, int, float, bool, list, dict)):
                        output = str(output)
                await websocket.send_json({
                    "type": "agent_update",
                    "agent": result.agent_name,
                    "task": result.task_type,
                    "status": "completed" if result.success else "failed",
                    "output": output if result.success else None,
                    "error": result.error if not result.success else None,
                    "duration": result.duration
                })
            
            # Step 1: Optimize query for niche biasing
            optimizer_task = AgentTask("optimizer", "optimize_query", query, priority=10)
            await on_start(optimizer_task)
            
            opt_results = await orchestrator.execute_tasks([optimizer_task])
            optimized_query = query
            if opt_results and opt_results[0].success:
                optimized_query = opt_results[0].output
                await on_complete(opt_results[0])
            
            # Step 2: Define and execute secondary tasks with optimized query
            tasks = [
                AgentTask("search", "web_search", optimized_query, priority=8),
                AgentTask("research", "research_topic", optimized_query, priority=7),
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
                    "query": query,
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
                    query=query,
                    results_json=json.dumps(ws_data),
                    niche_focus=getattr(orchestrator.agents.get("optimizer"), "niche_focus", None)
                )

            # Send final summary with session_id
            summary = {
                "type": "research_complete",
                "query": query,
                "total_time": total_time,
                "results_count": len(results),
                "session_id": session_id
            }
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
