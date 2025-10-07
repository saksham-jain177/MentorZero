"""
New Agent-based API Routes
Handles multi-agent orchestration and research workflows
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio

from agent.core.orchestrator import (
    AgentOrchestrator, 
    AgentTask, 
    ExecutionMode,
    SearchAgent,
    WritingAgent,
    OptimizationAgent
)
from agent.core.research_agent import ResearchAgent
from agent.core.hyde_retriever import HyDERetriever
from agent.core.capabilities import (
    CodeGenerationAgent,
    LearningAgent,
    AnalysisAgent,
    AutomationAgent,
    CreativeAgent
)

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

class ResearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    execution_time: float
    agents_used: List[str]
    mode: str
    system_stats: Dict[str, Any]

class AgentStatus(BaseModel):
    agent_name: str
    status: str
    current_task: Optional[str]
    tasks_completed: int

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
        
        # Step 1: Optimize the query
        tasks.append(AgentTask(
            agent_name="optimizer",
            task_type="optimize_query",
            input_data=request.query,
            priority=10
        ))
        
        # Step 2: Parallel search and research
        tasks.append(AgentTask(
            agent_name="search",
            task_type="web_search",
            input_data=request.query,
            priority=8
        ))
        
        if request.depth in ["standard", "deep"]:
            tasks.append(AgentTask(
                agent_name="research",
                task_type="research_topic",
                input_data=request.query,
                priority=8
            ))
        
        # Step 3: Summarize findings
        tasks.append(AgentTask(
            agent_name="writer",
            task_type="summarize",
            input_data="Combined research findings",
            priority=5,
            requires=["web_search"]
        ))
        
        # Execute tasks
        import time
        start_time = time.time()
        
        results = await orchestrator.execute_tasks(tasks, mode=execution_mode)
        
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
        
        return ResearchResponse(
            query=request.query,
            results=formatted_results,
            execution_time=execution_time,
            agents_used=agents_used,
            mode=request.mode,
            system_stats=orchestrator.resource_monitor.get_system_stats()
        )
        
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
async def research_websocket(websocket):
    """
    WebSocket endpoint for real-time research updates
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            if not query:
                await websocket.send_json({"error": "No query provided"})
                continue
            
            # Stream results as agents complete tasks
            tasks = [
                AgentTask("optimizer", "optimize_query", query, priority=10),
                AgentTask("search", "web_search", query, priority=8),
                AgentTask("research", "research_topic", query, priority=7),
            ]
            
            for task in tasks:
                # Execute single task
                result = await orchestrator._execute_single_task(task)
                
                # Send update to client
                await websocket.send_json({
                    "type": "agent_update",
                    "agent": task.agent_name,
                    "task": task.task_type,
                    "status": "completed" if result.success else "failed",
                    "output": result.output if result.success else None,
                    "error": result.error if not result.success else None,
                    "duration": result.duration
                })
                
                # Small delay for visual effect
                await asyncio.sleep(0.1)
            
            # Send final summary
            await websocket.send_json({
                "type": "research_complete",
                "query": query,
                "total_time": sum(r.duration for r in results)
            })
            
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

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

# Initialize agents on module load
asyncio.create_task(initialize_agents())
