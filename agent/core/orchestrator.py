"""
Multi-Agent Orchestrator
Manages specialized agents working in parallel or sequence
with intelligent resource management for local compute
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import json


class ExecutionMode(Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"  # Decides based on system resources


@dataclass
class AgentTask:
    agent_name: str
    task_type: str
    input_data: Any
    priority: int = 5
    max_duration: float = 30.0
    requires: List[str] = None  # Dependencies on other tasks


@dataclass
class TaskResult:
    agent_name: str
    task_type: str
    output: Any
    duration: float
    success: bool
    error: Optional[str] = None


class ResourceMonitor:
    """Monitors system resources to prevent overload"""
    
    def __init__(self, max_cpu_percent: float = 80, max_memory_percent: float = 70):
        self.max_cpu = max_cpu_percent
        self.max_memory = max_memory_percent
        self.active_agents = 0
        self.max_parallel_agents = self._calculate_max_agents()
    
    def _calculate_max_agents(self) -> int:
        """Calculate max parallel agents based on system specs"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative: 1 agent per 2 cores, max 4 for local machines
        max_by_cpu = max(1, cpu_count // 2)
        max_by_memory = max(1, int(memory_gb // 2))  # 1 agent per 2GB RAM
        
        return min(4, max_by_cpu, max_by_memory)
    
    def can_spawn_agent(self) -> bool:
        """Check if system can handle another agent"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        return (
            self.active_agents < self.max_parallel_agents and
            cpu_percent < self.max_cpu and
            memory_percent < self.max_memory
        )
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "active_agents": self.active_agents,
            "max_agents": self.max_parallel_agents
        }


class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents
    Handles parallel/sequential execution with resource management
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.agents: Dict[str, Any] = {}
        self.resource_monitor = ResourceMonitor()
        self.task_queue: List[AgentTask] = []
        self.results: Dict[str, TaskResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def register_agent(self, name: str, agent_instance: Any):
        """Register a specialized agent"""
        self.agents[name] = agent_instance
        print(f"[OK] Registered agent: {name}")
    
    async def execute_tasks(
        self, 
        tasks: List[AgentTask], 
        mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        on_task_start: Optional[Callable[[AgentTask], Any]] = None,
        on_task_complete: Optional[Callable[[TaskResult], Any]] = None
    ) -> List[TaskResult]:
        """
        Execute multiple tasks with specified mode
        
        Modes:
        - PARALLEL: All at once (if resources allow)
        - SEQUENTIAL: One by one
        - ADAPTIVE: Decides based on system load
        """
        
        # Sort by priority
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        if mode == ExecutionMode.ADAPTIVE:
            mode = self._decide_execution_mode(tasks)
            print(f"[Adaptive] Mode selected: {mode.value}")
        
        if mode == ExecutionMode.PARALLEL:
            return await self._execute_parallel(tasks, on_task_start, on_task_complete)
        else:
            return await self._execute_sequential(tasks, on_task_start, on_task_complete)
    
    def _decide_execution_mode(self, tasks: List[AgentTask]) -> ExecutionMode:
        """Intelligently decide execution mode based on system state"""
        stats = self.resource_monitor.get_system_stats()
        
        # If system is already loaded, go sequential
        if stats["cpu_percent"] > 60 or stats["memory_percent"] > 60:
            return ExecutionMode.SEQUENTIAL
        
        # If we have few tasks, parallel is fine
        if len(tasks) <= 2:
            return ExecutionMode.PARALLEL
        
        # Check task dependencies
        has_dependencies = any(task.requires for task in tasks)
        if has_dependencies:
            return ExecutionMode.SEQUENTIAL
        
        return ExecutionMode.PARALLEL
    
    async def _execute_parallel(
        self, 
        tasks: List[AgentTask],
        on_task_start: Optional[Callable[[AgentTask], Any]] = None,
        on_task_complete: Optional[Callable[[TaskResult], Any]] = None
    ) -> List[TaskResult]:
        """Execute tasks in parallel with resource throttling"""
        results = []
        
        # Group tasks into batches based on available resources
        batch_size = self.resource_monitor.max_parallel_agents
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            # Wait for resources if needed
            while not self.resource_monitor.can_spawn_agent():
                print("[wait] Waiting for resources...")
                await asyncio.sleep(1)
            
            # Execute batch in parallel
            batch_tasks = [self._execute_single_task(task, on_task_start, on_task_complete) for task in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for task, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    res = TaskResult(
                        agent_name=task.agent_name,
                        task_type=task.task_type,
                        output=None,
                        duration=0,
                        success=False,
                        error=str(result)
                    )
                    results.append(res)
                    if on_task_complete:
                        if asyncio.iscoroutinefunction(on_task_complete):
                            await on_task_complete(res)
                        else:
                            on_task_complete(res)
                else:
                    results.append(result)
        
        return results
    
    async def _execute_sequential(
        self, 
        tasks: List[AgentTask],
        on_task_start: Optional[Callable[[AgentTask], Any]] = None,
        on_task_complete: Optional[Callable[[TaskResult], Any]] = None
    ) -> List[TaskResult]:
        """Execute tasks one by one"""
        results = []
        
        for task in tasks:
            # Check dependencies
            if task.requires:
                skip = False
                for dep in task.requires:
                    if dep not in [r.task_type for r in results if r.success]:
                        print(f"[skip] Skipping {task.task_type}: dependency {dep} not met")
                        skip = True
                        break
                if skip: continue
            
            result = await self._execute_single_task(task, on_task_start, on_task_complete)
            results.append(result)
            
            # Small delay to prevent CPU spikes
            await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_single_task(
        self, 
        task: AgentTask,
        on_task_start: Optional[Callable[[AgentTask], Any]] = None,
        on_task_complete: Optional[Callable[[TaskResult], Any]] = None
    ) -> TaskResult:
        """Execute a single agent task with monitoring"""
        start_time = time.time()
        
        # Notify task start
        if on_task_start:
            if asyncio.iscoroutinefunction(on_task_start):
                await on_task_start(task)
            else:
                on_task_start(task)
        
        try:
            # Update resource monitor
            self.resource_monitor.active_agents += 1
            
            # Get the agent
            agent = self.agents.get(task.agent_name)
            if not agent:
                raise ValueError(f"Agent {task.agent_name} not registered")
            
            # Execute with timeout
            print(f">>> Starting: {task.agent_name}.{task.task_type}")
            
            # Call the appropriate method on the agent
            method = getattr(agent, task.task_type, None)
            if not method:
                raise ValueError(f"Agent {task.agent_name} has no method {task.task_type}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                method(task.input_data),
                timeout=task.max_duration
            )
            
            duration = time.time() - start_time
            print(f"[DONE] Completed: {task.agent_name}.{task.task_type} ({duration:.2f}s)")
            
            res = TaskResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                output=result,
                duration=duration,
                success=True
            )
            
            if on_task_complete:
                if asyncio.iscoroutinefunction(on_task_complete):
                    await on_task_complete(res)
                else:
                    on_task_complete(res)
                    
            return res
            
        except asyncio.TimeoutError:
            res = TaskResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                output=None,
                duration=task.max_duration,
                success=False,
                error=f"Timeout after {task.max_duration}s"
            )
            if on_task_complete:
                if asyncio.iscoroutinefunction(on_task_complete):
                    await on_task_complete(res)
                else:
                    on_task_complete(res)
            return res
        except Exception as e:
            res = TaskResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                output=None,
                duration=time.time() - start_time,
                success=False,
                error=str(e)
            )
            if on_task_complete:
                if asyncio.iscoroutinefunction(on_task_complete):
                    await on_task_complete(res)
                else:
                    on_task_complete(res)
            return res
        finally:
            self.resource_monitor.active_agents -= 1
    
    def get_execution_plan(self, tasks: List[AgentTask]) -> Dict:
        """Generate an execution plan showing how tasks will run"""
        plan = {
            "total_tasks": len(tasks),
            "system_stats": self.resource_monitor.get_system_stats(),
            "execution_mode": self._decide_execution_mode(tasks).value,
            "task_order": []
        }
        
        # Sort by priority and dependencies
        sorted_tasks = sorted(tasks, key=lambda x: x.priority, reverse=True)
        
        for task in sorted_tasks:
            plan["task_order"].append({
                "agent": task.agent_name,
                "task": task.task_type,
                "priority": task.priority,
                "requires": task.requires or []
            })
        
        return plan


# Example specialized agents
class SearchAgent:
    """Agent specialized in searching and research"""
    
    async def web_search(self, query: str) -> Dict:
        """Search the web for information"""
        await asyncio.sleep(1)  # Simulate API call
        return {
            "query": query,
            "results": [f"Result 1 for {query}", f"Result 2 for {query}"],
            "sources": ["web"]
        }
    
    async def deep_research(self, topic: str) -> Dict:
        """Deep research on a topic"""
        await asyncio.sleep(2)  # Simulate longer research
        return {
            "topic": topic,
            "findings": f"Comprehensive research on {topic}",
            "confidence": 0.85
        }


class WritingAgent:
    """Agent specialized in content generation"""
    
    async def summarize(self, content: str) -> str:
        """Summarize content"""
        await asyncio.sleep(0.5)
        return f"Summary of: {content[:50]}..."
    
    async def expand(self, outline: str) -> str:
        """Expand an outline into full content"""
        await asyncio.sleep(1)
        return f"Expanded version of: {outline}"


class OptimizationAgent:
    """Agent specialized in optimization and refinement"""
    
    async def optimize_query(self, query: str) -> str:
        """Optimize a search query"""
        await asyncio.sleep(0.3)
        return f"Optimized: {query}"
    
    async def improve_answer(self, answer: str) -> str:
        """Improve an answer's quality"""
        await asyncio.sleep(0.5)
        return f"Improved: {answer}"


# Demo usage
async def demo_orchestrator():
    """Demonstrate the orchestrator with multiple agents"""
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Register specialized agents
    orchestrator.register_agent("search", SearchAgent())
    orchestrator.register_agent("writer", WritingAgent())
    orchestrator.register_agent("optimizer", OptimizationAgent())
    
    # Define a complex multi-agent workflow
    tasks = [
        AgentTask(
            agent_name="optimizer",
            task_type="optimize_query",
            input_data="How does quantum computing work?",
            priority=10
        ),
        AgentTask(
            agent_name="search",
            task_type="web_search",
            input_data="quantum computing basics",
            priority=8,
            requires=["optimize_query"]
        ),
        AgentTask(
            agent_name="search",
            task_type="deep_research",
            input_data="quantum computing",
            priority=7
        ),
        AgentTask(
            agent_name="writer",
            task_type="summarize",
            input_data="Research findings",
            priority=5,
            requires=["web_search", "deep_research"]
        ),
    ]
    
    # Show execution plan
    plan = orchestrator.get_execution_plan(tasks)
    print("\nExecution Plan:")
    print(json.dumps(plan, indent=2))
    
    # Execute tasks
    print("\nExecuting tasks...")
    results = await orchestrator.execute_tasks(tasks, mode=ExecutionMode.ADAPTIVE)
    
    # Show results
    print("\nResults:")
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.agent_name}.{result.task_type}: {result.duration:.2f}s")
        if result.error:
            print(f"   Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(demo_orchestrator())
