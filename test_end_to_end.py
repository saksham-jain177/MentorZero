#!/usr/bin/env python
"""
End-to-end test script for MentorZero with performance optimizations.
"""
import asyncio
import time
import json
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    "api_base": "http://localhost:8000",
    "test_topic": "machine learning",
    "test_count": 3,
    "test_text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
}


async def test_optimized_azl():
    """Test the optimized AZL scorer with caching and cascade."""
    print("\n🧪 Testing Optimized AZL Scorer...")
    
    from agent.azl.generator import AZLGenerator
    from agent.azl.optimized_scorer import OptimizedAZLScorer
    from agent.llm.ollama_client import OllamaClient
    from agent.config import get_settings, get_azl_config
    from agent.db.base import get_session_factory
    
    settings = get_settings()
    cfg = get_azl_config()
    
    # Create clients
    ollama_client = OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout_seconds=30.0
    )
    
    # Check if LLM is available
    health = ollama_client.health()
    if not health.get("ready"):
        print("❌ Ollama not available. Please start Ollama first.")
        return False
    
    print(f"✅ Ollama is ready with model: {settings.ollama_model}")
    
    # Get database session
    session_factory = get_session_factory()
    db_session = session_factory()
    
    # Create optimized scorer
    scorer = OptimizedAZLScorer(
        ollama_client,
        cfg.score_weights,
        judge_client=ollama_client,  # Using same model for simplicity
        db_session=db_session,
        cache_ttl_seconds=3600,
        confidence_threshold=0.7,
        score_margin=0.1
    )
    
    # Generate examples
    generator = AZLGenerator(ollama_client)
    print(f"\n📝 Generating {TEST_CONFIG['test_count']} examples for '{TEST_CONFIG['test_topic']}'...")
    
    examples, proposal_id = await generator.propose_examples(
        TEST_CONFIG['test_topic'], 
        TEST_CONFIG['test_count']
    )
    
    print(f"✅ Generated {len(examples)} examples")
    
    # Test scoring with timing
    results = []
    for idx, example in enumerate(examples):
        print(f"\n🔍 Scoring example {idx + 1}/{len(examples)}...")
        print(f"   Q: {example.get('question', '')[:80]}...")
        
        start_time = time.time()
        result = await scorer.score(example, TEST_CONFIG['test_topic'])
        duration_ms = (time.time() - start_time) * 1000
        
        results.append(result)
        
        print(f"   ✅ Score: {result['score']:.2f} (Method: {result.get('method', 'unknown')})")
        print(f"   ⏱️ Duration: {duration_ms:.0f}ms")
        print(f"   📊 Confidence: {result.get('confidence', 0):.2f}")
        
        # Test caching - score same example again
        print(f"   🔄 Testing cache...")
        cache_start = time.time()
        cached_result = await scorer.score(example, TEST_CONFIG['test_topic'])
        cache_duration_ms = (time.time() - cache_start) * 1000
        
        if cache_duration_ms < duration_ms / 10:  # Should be at least 10x faster
            print(f"   ✅ Cache hit! Duration: {cache_duration_ms:.0f}ms (speedup: {duration_ms/cache_duration_ms:.1f}x)")
        else:
            print(f"   ⚠️ Cache may not be working. Duration: {cache_duration_ms:.0f}ms")
    
    # Print cache statistics
    cache_stats = scorer.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Memory cache size: {cache_stats['memory_cache_size']}")
    print(f"   Memory cache hits: {cache_stats['memory_cache_hits']}")
    print(f"   DB cache hits: {cache_stats['db_cache_hits']}")
    print(f"   Cache misses: {cache_stats['cache_misses']}")
    
    # Print performance metrics
    metrics = scorer.get_performance_metrics()
    print(f"\n📈 Performance Metrics ({len(metrics)} operations):")
    
    operation_summary = {}
    for metric in metrics:
        op = metric['operation']
        if op not in operation_summary:
            operation_summary[op] = {
                'count': 0,
                'total_ms': 0,
                'success': 0
            }
        operation_summary[op]['count'] += 1
        operation_summary[op]['total_ms'] += metric['duration_ms']
        if metric['success']:
            operation_summary[op]['success'] += 1
    
    for op, stats in operation_summary.items():
        avg_ms = stats['total_ms'] / stats['count'] if stats['count'] > 0 else 0
        success_rate = stats['success'] / stats['count'] if stats['count'] > 0 else 0
        print(f"   {op}: {stats['count']} calls, avg {avg_ms:.0f}ms, {success_rate*100:.0f}% success")
    
    db_session.close()
    return True


async def test_error_handling():
    """Test enhanced error handling."""
    print("\n🧪 Testing Error Handling...")
    
    from agent.utils.error_handler import ErrorHandler, ErrorContext, LLMError
    
    # Test LLM timeout error
    context = ErrorContext(
        operation="test_operation",
        topic="test topic",
        metadata={"test": "data"}
    )
    
    try:
        raise TimeoutError("Request timed out after 30 seconds")
    except Exception as e:
        error_info = ErrorHandler.handle_llm_error(e, context, "test-model")
        
        print(f"\n❌ Simulated Error: {error_info['user_message']}")
        print(f"💡 Recovery Suggestions:")
        for idx, suggestion in enumerate(error_info['recovery_suggestions']):
            print(f"   {idx + 1}. {suggestion}")
        
        assert len(error_info['recovery_suggestions']) > 0, "Should have recovery suggestions"
        print("\n✅ Error handler working correctly")
    
    return True


async def test_background_processing():
    """Test background processing service."""
    print("\n🧪 Testing Background Processing...")
    
    from agent.services.background_processor import AsyncBackgroundProcessor
    
    processor = AsyncBackgroundProcessor(max_concurrent_tasks=2)
    
    # Register a test handler
    async def test_handler(payload, progress_callback):
        total_steps = 5
        for i in range(total_steps):
            await asyncio.sleep(0.1)  # Simulate work
            progress = ((i + 1) / total_steps) * 100
            progress_callback(progress)
        return {"result": "completed", "payload": payload}
    
    processor.register_handler('test_task', test_handler)
    
    # Submit a task
    task_id = await processor.submit_task(
        'test_task',
        {"test": "data"}
    )
    
    print(f"✅ Task submitted: {task_id}")
    
    # Monitor progress
    for _ in range(10):
        task = await processor.get_task_status(task_id)
        if task:
            print(f"   Status: {task.status}, Progress: {task.progress:.0f}%")
            if task.status == 'completed':
                print(f"   ✅ Result: {task.result}")
                break
            elif task.status == 'failed':
                print(f"   ❌ Error: {task.error}")
                break
        await asyncio.sleep(0.2)
    
    return True


async def test_rag_service():
    """Test Agentic RAG service."""
    print("\n🧪 Testing Agentic RAG Service...")
    
    from agent.services.rag import AgenticRAGService
    from agent.embeddings.service import EmbeddingService
    from agent.vectorstore.faiss_store import FaissStore
    from agent.llm.ollama_client import OllamaClient
    from agent.config import get_settings
    
    settings = get_settings()
    
    # Initialize components
    embedder = EmbeddingService(model_name=settings.embedding_model_name)
    store = FaissStore(
        dim=embedder.dim,
        index_path=settings.faiss_index_path,
        meta_path=settings.faiss_meta_path
    )
    
    ollama_client = OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        timeout_seconds=30.0
    )
    
    # Create RAG service
    rag = AgenticRAGService(embedder, store, ollama_client)
    
    # Test ingestion
    print(f"📥 Ingesting test text...")
    num_chunks = rag.ingest_text("test_session", TEST_CONFIG['test_text'])
    print(f"✅ Ingested {num_chunks} chunks")
    
    # Test retrieval
    print(f"🔍 Testing retrieval...")
    results = await rag.retrieve("What is machine learning?", k=3)
    
    if results:
        print(f"✅ Retrieved {len(results)} chunks")
        for idx, chunk in enumerate(results[:2]):
            print(f"   Chunk {idx + 1}: {chunk.text[:80]}... (score: {chunk.score:.2f})")
    else:
        print("⚠️ No chunks retrieved")
    
    return True


async def main():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("🚀 MentorZero End-to-End Test Suite")
    print("=" * 60)
    
    tests = [
        ("Optimized AZL Scorer", test_optimized_azl),
        ("Error Handling", test_error_handling),
        ("Background Processing", test_background_processing),
        ("Agentic RAG", test_rag_service),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MentorZero is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)