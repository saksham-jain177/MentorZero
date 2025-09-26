"""
Background processing service for heavy evaluations.
"""
import asyncio
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
import threading
import uuid


@dataclass
class BackgroundTask:
    """Represents a background task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: str  # 'pending', 'processing', 'completed', 'failed'
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class BackgroundProcessor:
    """
    Service for processing heavy tasks in the background.
    
    Features:
    - Async task execution
    - Progress tracking
    - Result caching
    - Error handling with retries
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.tasks: Dict[str, BackgroundTask] = {}
        self.task_queue: Queue = Queue()
        self.workers: list[threading.Thread] = []
        self.running = False
        self._task_handlers: Dict[str, Callable] = {}
        
    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler for a specific task type."""
        self._task_handlers[task_type] = handler
        
    def start(self) -> None:
        """Start the background processor."""
        if self.running:
            return
            
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
            
    def stop(self) -> None:
        """Stop the background processor."""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)
        self.workers.clear()
        
    def submit_task(
        self, 
        task_type: str, 
        payload: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> str:
        """Submit a task for background processing."""
        if task_id is None:
            task_id = str(uuid.uuid4())
            
        task = BackgroundTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            status='pending'
        )
        
        self.tasks[task_id] = task
        self.task_queue.put(task_id)
        
        return task_id
        
    def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get the status of a task."""
        return self.tasks.get(task_id)
        
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed task."""
        task = self.tasks.get(task_id)
        if task and task.status == 'completed':
            return task.result
        return None
        
    def _worker_loop(self) -> None:
        """Worker loop for processing tasks."""
        while self.running:
            try:
                # Get task from queue with timeout
                task_id = self.task_queue.get(timeout=1)
                task = self.tasks.get(task_id)
                
                if not task:
                    continue
                    
                # Update task status
                task.status = 'processing'
                task.started_at = time.time()
                
                # Get handler for task type
                handler = self._task_handlers.get(task.task_type)
                if not handler:
                    task.status = 'failed'
                    task.error = f"No handler registered for task type: {task.task_type}"
                    task.completed_at = time.time()
                    continue
                    
                try:
                    # Execute handler
                    result = handler(task.payload, lambda p: self._update_progress(task_id, p))
                    
                    # Update task with result
                    task.status = 'completed'
                    task.result = result
                    task.progress = 100.0
                    task.completed_at = time.time()
                    
                except Exception as e:
                    task.status = 'failed'
                    task.error = str(e)
                    task.completed_at = time.time()
                    
            except:
                # Queue.get timeout, continue loop
                continue
                
    def _update_progress(self, task_id: str, progress: float) -> None:
        """Update task progress."""
        task = self.tasks.get(task_id)
        if task:
            task.progress = min(100.0, max(0.0, progress))


class AsyncBackgroundProcessor:
    """
    Async version of the background processor for use with FastAPI.
    """
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, BackgroundTask] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
    def register_handler(self, task_type: str, handler: Callable) -> None:
        """Register an async handler for a specific task type."""
        self._task_handlers[task_type] = handler
        
    async def submit_task(
        self, 
        task_type: str, 
        payload: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> str:
        """Submit a task for background processing."""
        if task_id is None:
            task_id = str(uuid.uuid4())
            
        task = BackgroundTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            status='pending'
        )
        
        self.tasks[task_id] = task
        
        # Start processing in background
        asyncio.create_task(self._process_task(task_id))
        
        return task_id
        
    async def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get the status of a task."""
        return self.tasks.get(task_id)
        
    async def wait_for_task(
        self, 
        task_id: str, 
        timeout: float = 300
    ) -> Optional[Dict[str, Any]]:
        """Wait for a task to complete and return its result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.tasks.get(task_id)
            if not task:
                return None
                
            if task.status == 'completed':
                return task.result
            elif task.status == 'failed':
                raise Exception(f"Task failed: {task.error}")
                
            await asyncio.sleep(0.5)
            
        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
        
    async def _process_task(self, task_id: str) -> None:
        """Process a task asynchronously."""
        async with self._semaphore:
            task = self.tasks.get(task_id)
            if not task:
                return
                
            # Update task status
            task.status = 'processing'
            task.started_at = time.time()
            
            # Get handler for task type
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                task.status = 'failed'
                task.error = f"No handler registered for task type: {task.task_type}"
                task.completed_at = time.time()
                return
                
            try:
                # Execute handler
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(
                        task.payload, 
                        lambda p: self._update_progress(task_id, p)
                    )
                else:
                    # Run sync handler in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        handler, 
                        task.payload,
                        lambda p: self._update_progress(task_id, p)
                    )
                
                # Update task with result
                task.status = 'completed'
                task.result = result
                task.progress = 100.0
                task.completed_at = time.time()
                
            except Exception as e:
                task.status = 'failed'
                task.error = str(e)
                task.completed_at = time.time()
                
    def _update_progress(self, task_id: str, progress: float) -> None:
        """Update task progress."""
        task = self.tasks.get(task_id)
        if task:
            task.progress = min(100.0, max(0.0, progress))


# Global instance for the application
background_processor = AsyncBackgroundProcessor(max_concurrent_tasks=5)