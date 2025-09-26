from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from agent.azl.scorer import AZLScorer
from agent.services.validators import AZLValidators
from agent.db.models import EvaluationCache, PerformanceMetrics
from agent.utils.error_handler import ErrorHandler, ErrorContext, AZLError, LLMError


class OptimizedAZLScorer(AZLScorer):
    """
    Optimized AZL scorer with smart caching and cascade logic.
    
    Features:
    - Hybrid caching (in-memory + SQLite)
    - Judge cascade (fast model first, reasoning only when needed)
    - Performance metrics tracking
    - Smart pre-filtering with confidence thresholds
    """

    def __init__(
        self, 
        ollama_client, 
        weights: Dict[str, float], 
        judge_client=None,
        db_session: Optional[Session] = None,
        cache_ttl_seconds: int = 3600,
        confidence_threshold: float = 0.7,
        score_margin: float = 0.1
    ):
        super().__init__(ollama_client, weights, judge_client)
        
        # Clients for cascade
        self.fast_client = ollama_client  # Fast model (llama3.1:8b)
        self.reasoning_client = judge_client or ollama_client  # Reasoning model (gpt-oss:20b)
        
        # Caching
        self.db_session = db_session
        self.cache_ttl_seconds = cache_ttl_seconds
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cascade logic
        self.confidence_threshold = confidence_threshold
        self.score_margin = score_margin  # Within margin of pass threshold triggers heavy judge
        
        # Performance tracking
        self._metrics: list[Dict[str, Any]] = []

    def _hash_example(self, example: Dict[str, str], topic: str = "") -> str:
        """Create a hash key for caching."""
        content = json.dumps({
            "question": example.get("question", ""),
            "answer": example.get("answer", ""),
            "topic": topic
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_from_memory_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from in-memory cache."""
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl_seconds:
                return cached["data"]
            else:
                # Expired, remove
                del self._memory_cache[cache_key]
        return None

    def _set_memory_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Set result in in-memory cache."""
        self._memory_cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }

    def _get_from_db_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from database cache."""
        if not self.db_session:
            return None
        
        try:
            cache_entry = self.db_session.query(EvaluationCache).filter(
                EvaluationCache.example_hash == cache_key,
                EvaluationCache.expires_at > datetime.now(timezone.utc)
            ).first()
            
            if cache_entry:
                return {
                    "score": cache_entry.reasoning_score or cache_entry.fast_score,
                    "confidence": cache_entry.confidence,
                    "method": cache_entry.method,
                    "checks": cache_entry.checks or {},
                    "judge": cache_entry.judge_result or {},
                    "passed": (cache_entry.reasoning_score or cache_entry.fast_score or 0.0) >= 0.75
                }
        except Exception as e:
            print(f"DB cache read error: {e}")
        
        return None

    def _set_db_cache(
        self, 
        cache_key: str, 
        topic: str, 
        data: Dict[str, Any]
    ) -> None:
        """Set result in database cache."""
        if not self.db_session:
            return
        
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl_seconds)
            
            # Remove existing entry if it exists
            self.db_session.query(EvaluationCache).filter(
                EvaluationCache.example_hash == cache_key
            ).delete()
            
            # Create new entry
            cache_entry = EvaluationCache(
                example_hash=cache_key,
                topic=topic,
                fast_score=data.get("score") if data.get("method") == "fast" else None,
                reasoning_score=data.get("score") if data.get("method") == "reasoning" else None,
                confidence=data.get("confidence", 0.0),
                method=data.get("method", "fast"),
                checks=data.get("checks"),
                judge_result=data.get("judge"),
                expires_at=expires_at
            )
            
            self.db_session.add(cache_entry)
            self.db_session.commit()
        except Exception as e:
            print(f"DB cache write error: {e}")
            if self.db_session:
                self.db_session.rollback()

    def _record_metric(
        self, 
        operation: str, 
        duration_ms: float, 
        success: bool, 
        model_used: str = "", 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record performance metric."""
        metric = {
            "metric_type": "azl_evaluation",
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "model_used": model_used,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        self._metrics.append(metric)
        
        # Persist to database
        if self.db_session:
            try:
                db_metric = PerformanceMetrics(
                    metric_type=metric["metric_type"],
                    operation=metric["operation"],
                    duration_ms=metric["duration_ms"],
                    success=metric["success"],
                    model_used=metric["model_used"],
                    extra_data=metric["metadata"]
                )
                self.db_session.add(db_metric)
                self.db_session.commit()
            except Exception as e:
                print(f"Metrics recording error: {e}")
                if self.db_session:
                    self.db_session.rollback()

    async def _fast_evaluate(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Fast evaluation using the main model without LLM-as-a-judge."""
        start_time = time.time()
        
        try:
            # Use existing validators but with fast model
            validators = AZLValidators(self.fast_client)
            checks = await validators.validate_all(example)
            
            # Calculate score without LLM judge
            passed_checks = sum(1 for check in checks.values() if check.get("passed", False))
            total_checks = len(checks)
            
            score = passed_checks / total_checks if total_checks > 0 else 0.0
            
            # Calculate confidence based on check consistency
            confidence = self._calculate_confidence(checks)
            
            duration_ms = (time.time() - start_time) * 1000
            self._record_metric("fast_evaluate", duration_ms, True, "fast_model")
            
            from agent.config import get_settings
            s = get_settings()
            pass_threshold = float(getattr(s, 'azl_threshold', 0.75) or 0.75)
            return {
                "score": score,
                "checks": checks,
                "judge": {"correctness": score, "clarity": score, "usefulness": score, "rationale": "Fast evaluation"},
                "confidence": confidence,
                "method": "fast",
                "passed": score >= pass_threshold
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_metric("fast_evaluate", duration_ms, False, "fast_model", {"error": str(e)})
            
            # Enhanced error handling
            context = ErrorContext(
                operation="azl_fast_evaluation",
                topic=getattr(self, '_current_topic', ''),
                metadata={"model": "fast_model", "duration_ms": duration_ms}
            )
            error_info = ErrorHandler.handle_azl_error(e, context, "evaluation")
            raise AZLError(
                error_info["message"], 
                context, 
                error_info["recovery_suggestions"],
                error_info["user_message"]
            )

    async def _reasoning_evaluate(self, example: Dict[str, str]) -> Dict[str, Any]:
        """Full reasoning evaluation using the reasoning model with LLM-as-a-judge."""
        start_time = time.time()
        
        try:
            # Use the original scorer logic with reasoning model
            validators = AZLValidators(self.reasoning_client)
            checks = await validators.validate_all(example)
            
            # Convert boolean checks into scores 0/1
            check_scores: Dict[str, float] = {}
            for name, result in checks.items():
                check_scores[name] = 1.0 if result.get("passed", False) else 0.0
            
            # LLM-as-a-judge with reasoning model
            judge = await self._judge_with_client(example, self.reasoning_client)
            
            # Weighted aggregate
            total = 0.0
            weight_sum = 0.0
            for name, val in check_scores.items():
                w = float(self.weights.get(name, 0.0))
                total += w * val
                weight_sum += w
            
            judge_weight = float(self.weights.get("judge", 0.0))
            judge_avg = (judge["correctness"] + judge["clarity"] + judge["usefulness"]) / 3.0
            total += judge_weight * judge_avg
            weight_sum += judge_weight
            
            final_score = total / max(1e-9, weight_sum)
            
            duration_ms = (time.time() - start_time) * 1000
            self._record_metric("reasoning_evaluate", duration_ms, True, "reasoning_model")
            
            from agent.config import get_settings
            s = get_settings()
            pass_threshold = float(getattr(s, 'azl_threshold', 0.75) or 0.75)
            return {
                "score": final_score,
                "checks": checks,
                "judge": judge,
                "confidence": 0.95,  # High confidence for reasoning
                "method": "reasoning",
                "passed": final_score >= pass_threshold
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_metric("reasoning_evaluate", duration_ms, False, "reasoning_model", {"error": str(e)})
            raise

    def _calculate_confidence(self, checks: Dict[str, Any]) -> float:
        """Calculate confidence based on check consistency."""
        passed_count = sum(1 for check in checks.values() if check.get("passed", False))
        total_count = len(checks)
        
        if total_count == 0:
            return 0.0
        
        # Higher confidence when checks are more consistent (all pass or all fail)
        consistency = abs(passed_count - total_count/2) / (total_count/2)
        base_confidence = passed_count / total_count
        
        return min(1.0, base_confidence + consistency * 0.3)

    async def _judge_with_client(self, example: Dict[str, str], client) -> Dict[str, float]:
        """Judge with specified client (for reasoning evaluation)."""
        system_prompt = (
            "You are a strict educational quality judge. Score a Q/A pair for tutoring use. "
            "Output ONLY JSON with fields: correctness (0..1), clarity (0..1), usefulness (0..1), rationale (string)."
        )
        prompt = (
            f"Question: {example.get('question','')}\n"
            f"Answer: {example.get('answer','')}\n\n"
            "Return compact JSON."
        )
        
        try:
            resp = await client.send(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
            text = resp.get("text", "{}")
            import re
            m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
            payload = m.group(1) if m else text
            obj = json.loads(payload)
            c = float(obj.get("correctness", 0))
            l = float(obj.get("clarity", 0))
            u = float(obj.get("usefulness", 0))
            rat = str(obj.get("rationale", ""))
            
            # clamp
            def clamp01(x: float) -> float:
                return max(0.0, min(1.0, x))
            return {"correctness": clamp01(c), "clarity": clamp01(l), "usefulness": clamp01(u), "rationale": rat}
        except Exception as e:
            return {"correctness": 0.0, "clarity": 0.0, "usefulness": 0.0, "rationale": f"judge_error: {str(e)}"}

    async def score(self, example: Dict[str, str], topic: str = "") -> Dict[str, Any]:
        """
        Main scoring method with smart caching and cascade logic.
        
        Flow:
        1. Check memory cache
        2. Check database cache
        3. Fast evaluation first
        4. Reasoning evaluation only if needed
        5. Cache results
        """
        start_time = time.time()
        cache_key = self._hash_example(example, topic)
        self._current_topic = topic  # Store for error context
        
        try:
            # Check memory cache first
            cached_result = self._get_from_memory_cache(cache_key)
            if cached_result:
                self._record_metric("cache_hit_memory", (time.time() - start_time) * 1000, True)
                return cached_result
            
            # Check database cache
            cached_result = self._get_from_db_cache(cache_key)
            if cached_result:
                # Also store in memory cache
                self._set_memory_cache(cache_key, cached_result)
                self._record_metric("cache_hit_db", (time.time() - start_time) * 1000, True)
                return cached_result
            
            # No cache hit, evaluate
            self._record_metric("cache_miss", (time.time() - start_time) * 1000, True)
            
            # Fast evaluation first (1-2 seconds)
            fast_result = await self._fast_evaluate(example)
            
            # Decision logic for cascade
            from agent.config import get_settings
            s = get_settings()
            pass_threshold = float(getattr(s, 'azl_threshold', 0.75) or 0.75)
            should_use_reasoning = (
                # Low confidence in fast result
                fast_result["confidence"] < self.confidence_threshold or
                # Score is within margin of pass threshold (ambiguous case)
                abs(fast_result["score"] - pass_threshold) <= self.score_margin
            )
            
            final_result = fast_result
            
            if should_use_reasoning and self.reasoning_client != self.fast_client:
                # Use reasoning model for uncertain cases
                try:
                    reasoning_result = await self._reasoning_evaluate(example)
                    final_result = reasoning_result
                except Exception as e:
                    print(f"Reasoning evaluation failed, using fast result: {e}")
                    # Fall back to fast result
            
            # Cache the result
            self._set_memory_cache(cache_key, final_result)
            self._set_db_cache(cache_key, topic, final_result)
            
            total_duration_ms = (time.time() - start_time) * 1000
            self._record_metric("score_complete", total_duration_ms, True, final_result.get("method", "unknown"))
            
            return final_result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_metric("score_complete", duration_ms, False, "unknown", {"error": str(e)})
            raise

    def get_performance_metrics(self) -> list[Dict[str, Any]]:
        """Get collected performance metrics."""
        return self._metrics.copy()

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache."""
        self._memory_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_hits": len([m for m in self._metrics if m["operation"] == "cache_hit_memory"]),
            "db_cache_hits": len([m for m in self._metrics if m["operation"] == "cache_hit_db"]),
            "cache_misses": len([m for m in self._metrics if m["operation"] == "cache_miss"])
        }