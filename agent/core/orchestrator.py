"""
Multi-Agent Orchestrator
Manages specialized agents working in parallel or sequence
with intelligent resource management for local compute
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import time

logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor
import psutil  # type: ignore[import-untyped]
import json
from datetime import datetime


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
    requires: Optional[List[str]] = None  # Dependencies on other tasks
    level: int = 0 # Hierarchical depth (Stage 25)


@dataclass
class TaskResult:
    agent_name: str
    task_type: str
    output: Any
    duration: float
    success: bool
    error: Optional[str] = None
    level: int = 0 # Hierarchical depth (Stage 25)


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
        logger.info(f"Registered agent: {name}")
    
    async def execute_tasks(
        self, 
        tasks: List[AgentTask], 
        mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        on_task_start: Optional[Callable[[AgentTask], Any]] = None,
        on_task_complete: Optional[Callable[[TaskResult], Any]] = None
    ) -> List[TaskResult]:
        """
        Execute multiple tasks with specified mode
        """
        self.task_queue = list(tasks)
        self.results = {}
        
        # Sort initial queue by priority
        self.task_queue.sort(key=lambda x: x.priority, reverse=True)
        
        if mode == ExecutionMode.ADAPTIVE:
            mode = self._decide_execution_mode(self.task_queue)
            print(f"[Adaptive] Mode selected: {mode.value}")
        
        # Process the queue until empty (supports dynamic spawning)
        final_results = []
        while self.task_queue:
            current_tasks = []
            
            if mode == ExecutionMode.PARALLEL:
                # Take a batch of tasks that can run in parallel
                batch_size = self.resource_monitor.max_parallel_agents
                current_tasks = self.task_queue[:batch_size]
                self.task_queue = self.task_queue[batch_size:]
                
                batch_results = await self._execute_parallel(current_tasks, on_task_start, on_task_complete)
                final_results.extend(batch_results)
            else:
                # Sequential
                task = self.task_queue.pop(0)
                result = await self._execute_sequential([task], on_task_start, on_task_complete)
                final_results.extend(result)
                
            # Allow for short break to let agents potentially spawn tasks via callbacks or direct refs
            # In a real system, we'd use an event loop or a more robust shared queue
            await asyncio.sleep(0.1)
            
        return final_results

    def spawn_task(self, task: AgentTask):
        """Allows agents to dynamically add tasks to the orchestrator"""
        logger.info(f"Dynamically spawned task: {task.agent_name}.{task.task_type}")
        # Insert at front if high priority, else append
        if task.priority > 5:
            self.task_queue.insert(0, task)
        else:
            self.task_queue.append(task)
    
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
            batch = tasks[i:i + batch_size]  # type: ignore
            
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
                        error=str(result),
                        level=task.level
                    )
                    results.append(res)
                    if on_task_complete:
                        if asyncio.iscoroutinefunction(on_task_complete):
                            await on_task_complete(res)
                        else:
                            on_task_complete(res)
                else:
                    if isinstance(result, TaskResult):
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
                for dep in task.requires:  # type: ignore
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
            
            # Execute with timeout and handle dict unpacking for complex inputs
            settings = get_settings()
            timeout = task.max_duration if task.max_duration else settings.task_global_timeout
            
            if isinstance(task.input_data, dict):
                result = await asyncio.wait_for(
                    method(**task.input_data),
                    timeout=timeout
                )
            else:
                result = await asyncio.wait_for(
                    method(task.input_data),
                    timeout=timeout
                )
            
            duration = time.time() - start_time
            print(f"[DONE] Completed: {task.agent_name}.{task.task_type} ({duration:.2f}s)")
            
            res = TaskResult(
                agent_name=task.agent_name,
                task_type=task.task_type,
                output=result,
                duration=duration,
                success=True,
                level=task.level
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
                error=f"Timeout after {task.max_duration}s",
                level=task.level
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
        
        # Fallback return to satisfy static analysis
        return TaskResult(
            agent_name=task.agent_name,
            task_type=task.task_type,
            output=None,
            duration=time.time() - start_time,
            success=False,
            error="Unknown execution error"
        )
    
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
            plan["task_order"].append({  # type: ignore
                "agent": task.agent_name,
                "task": task.task_type,
                "priority": task.priority,
                "requires": task.requires or []
            })
        
        return plan


class SearchAgent:
    """Specialized agent for web searching"""
    
    def __init__(self, llm_client=None):
        from agent.core.research_agent import WebSearchTool
        self.web_search = WebSearchTool()
        self.llm = llm_client
    
    async def web_search(self, query: str) -> List[Dict]:
        """Perform a web search"""
        return await self.web_search.search(query)


class WritingAgent:
    """Specialized agent for content generation and summarization"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def summarize(self, content: Any) -> str:
        """Summarize research findings with strict citations (Compliance Stage 21)"""
        if not self.llm:
            return f"Summary of {len(str(content))} bytes: [LLM Required]"
            
        prompt = f"""Summarize the following research data into a clear, professional report.
        
### RULES:
1. **Strict Citations**: Every factual claim MUST be followed by a citation like [Source URL] or [Document Name].
2. **Structure**: Use headers, bullet points, and a "Final Conclusion" section.
3. **No Hallucinations**: Do not add information not present in the provided context.

Research Data:
{content}"""
        return await self.llm.send_prompt(prompt)


class OptimizationAgent:
    """Specialized agent for query and research optimization"""
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    async def optimize(self, query: str) -> str:
        """Optimize a research query"""
        if not self.llm:
            return query
            
        prompt = f"Optimize this research query for better web search results: '{query}'. Return only the optimized query."
        return await self.llm.send_prompt(prompt)
