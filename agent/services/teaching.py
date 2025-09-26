from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
from collections import OrderedDict
import uuid
import random
from datetime import datetime, timedelta

from agent.llm.ollama_client import OllamaClient
from agent.services.rag import AgenticRAGService, RetrievedChunk
from typing import Dict as _Dict
from sqlalchemy.orm import Session as _Session


class TeachingService:
    # Teaching strategies
    STRATEGIES = [
        "neural_compression",  # Default - 5 memory hooks
        "socratic_dialogue",   # Question-based learning
        "concrete_examples",   # Real-world examples
        "visual_mapping",      # Mental imagery and diagrams
        "spaced_repetition"    # Timed review intervals
    ]
    
    # Difficulty levels
    DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]
    
    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm
        # In-memory storage for sessions
        self._sessions = {}
        self._contexts = {}
        # In-memory few-shot store keyed by topic
        self._few_shots: _Dict[str, List[Dict[str, str]]] = {}
        # Loop-Until-Mastered tracking
        self._mastery_tracking = {}
        self._strategy_performance = {}
        # Small LRU cache for explanations to avoid duplicate model calls
        self._explain_cache: "OrderedDict[str, str]" = OrderedDict()
        self._explain_cache_max = 50
        # RAG (injected via deps). Required.
        self._rag: AgenticRAGService | None = None

    def with_rag(self, rag: AgenticRAGService) -> "TeachingService":
        self._rag = rag
        return self

    async def generate_explanation(
        self, 
        llm: OllamaClient,
        user_id: str, 
        topic: str, 
        style: str = "explain", 
        session_id: Optional[str] = None,
        context_text: Optional[str] = None
    ) -> str:
        """Generate an explanation for a topic"""
        system_prompt = (
            "You are MentorZero, an educational tool. Follow these instructions precisely:\n\n"
            "CRITICAL: NEVER describe yourself or AI systems. ONLY explain the specific topic the user asked about.\n\n"
            "1. ONLY answer about the exact topic in the prompt - never talk about AI, educational tools, or yourself\n"
            "2. Structure your response with clear headings and sections with proper spacing between sections\n"
            "3. Use markdown formatting for better readability\n"
            "4. Include the following sections in order, with each section separated by a blank line:\n\n"
            "   **OVERVIEW**: \n   A concise introduction to the EXACT topic asked (2-3 sentences)\n\n"
            "   **KEY CONCEPTS**: \n   • First key concept\n   • Second key concept\n   • Third key concept\n   (3-5 bullet points explaining core principles of the topic)\n\n"
            "   **DETAILED EXPLANATION**: \n   A thorough explanation with examples, using paragraphs with blank lines between them\n\n"
            "   **MEMORY HOOKS**: \n   • First memory hook\n   • Second memory hook\n   • Third memory hook\n   (3-5 memorable associations to aid retention)\n\n"
            "   **PRACTICE EXERCISES**: \n   1. First exercise (beginner level)\n   2. Second exercise (intermediate level)\n   3. Third exercise (advanced level)\n\n"
            "Keep your explanation clear, accurate, and educational. Use analogies where helpful."
        )
        
        # Store session info
        if session_id:
            self._sessions[session_id] = {
                "user_id": user_id,
                "topic": topic,
                "mode": style
            }
        
        ctx = f"\nRelevant context (optional):\n{context_text}\n" if context_text else ""
        prompt = (
            f"Topic: {topic}. Style: {style}." + ctx +
            " Provide a comprehensive but structured explanation about this SPECIFIC TOPIC ONLY. " +
            "DO NOT describe yourself or AI systems in general. Focus exclusively on explaining " +
            f"'{topic}' following the format in the system prompt."
        )
        # Append few-shot examples if available
        few = self._few_shots.get(topic, [])[:2]
        if few:
            examples_block = "\n\nExamples to mirror style (do not copy text):\n" + "\n".join(
                [f"Q: {e.get('question','').strip()}\nA: {e.get('answer','').strip()}" for e in few]
            )
            prompt += examples_block
        
        try:
            # Cache key uses topic, style, and context hash (if any)
            ctx_hash = hashlib.sha1((context_text or "").encode("utf-8")).hexdigest()[:8]
            cache_key = f"explain::{topic}::{style}::{ctx_hash}"
            if cache_key in self._explain_cache:
                # Move to end (LRU)
                text = self._explain_cache.pop(cache_key)
                self._explain_cache[cache_key] = text
                return text

            resp = await llm.send(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
            text = resp.get("text", "")
            # Update LRU cache
            self._explain_cache[cache_key] = text
            if len(self._explain_cache) > self._explain_cache_max:
                self._explain_cache.popitem(last=False)
            return text
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return f"Failed to generate explanation: {str(e)}"

    async def generate_quiz(
        self, 
        llm: OllamaClient,
        topic: str, 
        difficulty: str = "beginner", 
        context_text: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Generate quiz questions for a topic"""
        sys_prompt = (
            "You are MentorZero. Create a structured quiz following these guidelines:\n\n"
            "CRITICAL: NEVER describe yourself or AI systems. ONLY create questions about the specific topic.\n\n"
            "1. Generate exactly 3 questions of increasing difficulty about the SPECIFIC TOPIC only\n"
            "2. For each question, provide:\n"
            "   - A clear, specific question about the topic\n"
            "   - The correct answer\n"
            "   - A brief explanation of why this answer is correct\n"
            "   - 3 incorrect options (for multiple choice)\n\n"
            "Format your response as a JSON array with objects containing 'question', 'answer', 'explanation', and 'options'"
        )
        
        ctx = f"\nRelevant context (optional):\n{context_text}\n" if context_text else ""
        prompt = (
            f"Topic: {topic}; difficulty: {difficulty}." + ctx +
            f" Create a structured quiz about '{topic}' ONLY. DO NOT include any questions about AI or educational tools. " +
            "Focus exclusively on the specified topic following the format in the system prompt."
        )
        
        try:
            resp = await llm.send(prompt=prompt, system_prompt=sys_prompt, temperature=0.0)
            text = resp.get("text", "")
            
            # Parse JSON response
            try:
                # Try to extract JSON from the text if it's not pure JSON
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    data = json.loads(text)
                
                if isinstance(data, list):
                    return data
                return []
            except:
                print(f"Failed to parse quiz JSON: {text[:100]}...")
                return []
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return []

    async def generate_feedback(
        self, 
        llm: OllamaClient,
        session_id: str, 
        user_answer: str,
        context_text: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Generate feedback for a user answer and evaluate correctness"""
        session_info = self.get_session_info(session_id)
        if not session_info:
            return "Session not found", False
        
        topic = session_info.get("topic", "")
        mode = session_info.get("mode", "explain")
        
        # Get current strategy and difficulty based on mastery tracking
        tracking = self._mastery_tracking.get(session_id, {})
        current_strategy = tracking.get("current_strategy", "neural_compression")
        current_difficulty = tracking.get("current_difficulty", "beginner")
        
        # Update strategy-specific system prompt
        system_prompt = self._get_strategy_prompt(current_strategy)
        
        ctx = f"\nRelevant context (optional):\n{context_text}\n" if context_text else ""
        prompt = (
            f"Topic: {topic}. Mode: {mode}. Difficulty: {current_difficulty}. " + 
            f"Student answer: {user_answer}" + ctx +
            " Provide structured feedback following the format in the system prompt. " +
            "At the end, include a single line with 'CORRECT: true' or 'CORRECT: false' " +
            "to indicate if the answer is fundamentally correct (even if incomplete)."
        )
        
        try:
            resp = await llm.send(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
            feedback_text = resp.get("text", "")
            
            # Extract correctness assessment
            is_correct = "CORRECT: true" in feedback_text.lower()
            
            # Update mastery tracking
            self._update_mastery_tracking(session_id, is_correct)
            
            # Remove the CORRECT line from the feedback
            feedback_text = feedback_text.replace("CORRECT: true", "").replace("CORRECT: false", "")
            
            return feedback_text, is_correct
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return f"Failed to generate feedback: {str(e)}", False

    async def generate_next_item(
        self, 
        llm: OllamaClient,
        session_id: str,
        context_text: Optional[str] = None
    ) -> str:
        """Generate the next learning item"""
        session_info = self.get_session_info(session_id)
        if not session_info:
            return "Session not found"
        
        topic = session_info.get("topic", "")
        mode = session_info.get("mode", "explain")
        
        # Get current strategy and difficulty based on mastery tracking
        tracking = self._mastery_tracking.get(session_id, {})
        current_strategy = tracking.get("current_strategy", "neural_compression")
        current_difficulty = tracking.get("current_difficulty", "beginner")
        mastery_level = tracking.get("mastery_level", 0)
        
        # Determine if we should repeat a concept based on mastery level
        should_repeat = mastery_level < 2 and random.random() < 0.3
        repeat_note = ""
        
        if should_repeat:
            repeat_note = "The learner needs reinforcement on the core concepts. Revisit a fundamental aspect of this topic."
        
        if mode == "quiz":
            system_prompt = (
                f"You are MentorZero using the {current_strategy} teaching strategy. " +
                f"Generate a {current_difficulty}-level question about the topic.\n"
                "Make the question specific, thought-provoking, and designed to test understanding.\n"
                "Format: A clear, direct question without providing the answer."
            )
        else:
            system_prompt = (
                f"You are MentorZero using the {current_strategy} teaching strategy. " +
                f"Generate a {current_difficulty}-level follow-up to deepen understanding.\n"
                "The prompt should encourage critical thinking and application of concepts.\n"
                "Format: A clear, direct question that builds on previous knowledge."
            )
        
        ctx = f"\nRelevant context (optional):\n{context_text}\n" if context_text else ""
        prompt = (
            f"Topic: {topic}. Mode: {mode}. Difficulty: {current_difficulty}." + ctx +
            f" {repeat_note} Generate the next learning item to deepen understanding."
        )
        
        try:
            resp = await llm.send(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
            return resp.get("text", "")
        except Exception as e:
            print(f"Error generating next item: {e}")
            return f"Failed to generate next item: {str(e)}"

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        return self._sessions.get(session_id)

    def process_text(self, session_id: str, text: str) -> int:
        """Process text for RAG: chunk -> embed -> index. Requires RAG to be configured."""
        # Keep last raw text in memory for debugging
        self._contexts[session_id] = text
        if self._rag is None:
            raise RuntimeError("Agentic RAG not configured")
        return self._rag.ingest_text(session_id, text)

    async def retrieve_context(self, session_id: str, topic: str) -> str:
        """Retrieve top-k context via RAG and return formatted citations. Requires RAG."""
        if self._rag is None:
            raise RuntimeError("Agentic RAG not configured")
        q = (topic or self._sessions.get(session_id, {}).get("topic", "")).strip()
        # Allow frontend to control k and MMR via window.MZ_RAG (passed through request context later)
        from agent.config import get_settings
        s = get_settings()
        chunks: List[RetrievedChunk] = await self._rag.retrieve(
            q,
            k=int(getattr(s, 'rag_top_k', 5) or 5),
            lambda_mult=float(getattr(s, 'rag_lambda_mult', 0.65) or 0.65),
        )
        return self._rag.format_citations(chunks) if chunks else ""

    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Get progress for a session"""
        # Get mastery tracking data
        tracking = self._mastery_tracking.get(session_id, {})
        session_info = self.get_session_info(session_id) or {}
        topic = session_info.get("topic", "")
        
        # Calculate metrics
        attempts = tracking.get("attempts", 0)
        correct = tracking.get("correct_answers", 0)
        accuracy = correct / max(attempts, 1) if attempts > 0 else 0.0
        streak = tracking.get("current_streak", 0)
        mastery_level = tracking.get("mastery_level", 0)
        current_difficulty = tracking.get("current_difficulty", "beginner")
        
        # Topics in progress
        topics_in_progress = []
        if topic:
            topics_in_progress.append({
                "topic": topic,
                "mastery_level": mastery_level,
                "current_difficulty": current_difficulty,
                "accuracy": accuracy
            })
        
        return {
            "accuracy": round(accuracy * 100, 1),
            "streak": streak,
            "topics_completed": tracking.get("topics_completed", 0),
            "topics_in_progress": topics_in_progress,
            "mastery_level": mastery_level,
            "current_difficulty": current_difficulty,
            "current_strategy": tracking.get("current_strategy", "neural_compression"),
            "attempts": attempts,
            "correct_answers": correct
        }
        
    def store_synthetic_example(
        self,
        example: Dict[str, str],
        message: str,
        topic: Optional[str] = None,
        db: Optional[_Session] = None,
        score: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a synthetic example and cache for few-shot use."""
        try:
            # Cache for few-shot generation
            t = (topic or "_generic").strip() or "_generic"
            self._few_shots.setdefault(t, [])
            # Keep a small rolling window per topic
            self._few_shots[t].append({"question": example.get("question", ""), "answer": example.get("answer", "")})
            if len(self._few_shots[t]) > 20:
                self._few_shots[t] = self._few_shots[t][-20:]

            # Optionally persist to DB
            if db is not None:
                try:
                    from agent.db.crud import create_synthetic_example
                    from uuid import uuid4 as _uuid4
                    payload = {
                        "question": example.get("question", ""),
                        "answer": example.get("answer", ""),
                        "topic": topic,
                        "message": message,
                        "score": score,
                        "meta": meta or {},
                    }
                    create_synthetic_example(
                        db=db,
                        example_id=str(_uuid4()),
                        source="azl",
                        payload=payload,
                        validator_passed=True,
                        verified_by=None,
                    )
                    db.commit()
                except Exception:
                    # Non-fatal: still keep in memory
                    pass
            return True
        except Exception:
            return False
        
    def _get_strategy_prompt(self, strategy: str) -> str:
        """Get the system prompt for a specific teaching strategy"""
        prompts = {
            "neural_compression": (
                "You are MentorZero using Neural Compression Protocol. Structure your feedback as follows:\n\n"
                "1. **ASSESSMENT**: Brief evaluation of the answer's accuracy\n"
                "2. **STRENGTHS**: What was correct or well-explained\n"
                "3. **AREAS FOR IMPROVEMENT**: Concepts that need clarification\n"
                "4. **MEMORY HOOKS**: 2-3 memorable associations to reinforce key concepts\n"
                "5. **CORRECT ANSWER**: The complete, accurate answer\n\n"
                "Be encouraging but thorough in your feedback."
            ),
            "socratic_dialogue": (
                "You are MentorZero using Socratic Dialogue. Structure your feedback as follows:\n\n"
                "1. **REFLECTION**: Ask the learner to reflect on their answer\n"
                "2. **GUIDING QUESTIONS**: 2-3 questions to lead the learner to insights\n"
                "3. **CONNECTING CONCEPTS**: Help the learner connect ideas\n"
                "4. **CORRECT APPROACH**: Guide toward the right method without giving the answer directly\n\n"
                "Focus on leading the learner to discover answers through questioning."
            ),
            "concrete_examples": (
                "You are MentorZero using Concrete Examples. Structure your feedback as follows:\n\n"
                "1. **ASSESSMENT**: Brief evaluation of the answer's accuracy\n"
                "2. **REAL-WORLD EXAMPLE**: A practical application of the concept\n"
                "3. **COUNTEREXAMPLE**: An example showing why misconceptions are incorrect\n"
                "4. **CORRECT ANSWER**: The complete answer with another practical example\n\n"
                "Use tangible, relatable examples to illustrate concepts."
            ),
            "visual_mapping": (
                "You are MentorZero using Visual Mapping. Structure your feedback as follows:\n\n"
                "1. **ASSESSMENT**: Brief evaluation of the answer's accuracy\n"
                "2. **MENTAL IMAGE**: A visual metaphor or analogy for the concept\n"
                "3. **STRUCTURAL MAPPING**: Describe how concepts connect visually\n"
                "4. **CORRECT ANSWER**: The complete answer with visual organization\n\n"
                "Help the learner create mental images and visual relationships between concepts."
            ),
            "spaced_repetition": (
                "You are MentorZero using Spaced Repetition. Structure your feedback as follows:\n\n"
                "1. **QUICK REVIEW**: Recap previously covered points\n"
                "2. **ASSESSMENT**: Evaluation of the current answer\n"
                "3. **CONNECTION**: How this connects to previous material\n"
                "4. **CORRECT ANSWER**: The complete answer\n"
                "5. **RECALL PROMPT**: A question to help remember this in the future\n\n"
                "Emphasize connections to previously learned material and future recall."
            )
        }
        
        return prompts.get(strategy, prompts["neural_compression"])
    
    def _initialize_mastery_tracking(self, session_id: str, user_id: str, topic: str) -> None:
        """Initialize mastery tracking for a new session"""
        if session_id not in self._mastery_tracking:
            # Get user's previous strategy performance if available
            user_strategies = self._strategy_performance.get(user_id, {})
            
            # Select best strategy or default if no history
            best_strategy = "neural_compression"
            if user_strategies:
                best_strategy = max(user_strategies.items(), key=lambda x: x[1]["success_rate"])[0]
            
            self._mastery_tracking[session_id] = {
                "user_id": user_id,
                "topic": topic,
                "attempts": 0,
                "correct_answers": 0,
                "current_streak": 0,
                "mastery_level": 0,  # 0-10 scale
                "current_difficulty": "beginner",
                "current_strategy": best_strategy,
                "last_strategy_change": datetime.now(),
                "topics_completed": 0,
                "retry_count": 0,
                "last_question": None
            }
    
    def _update_mastery_tracking(self, session_id: str, is_correct: bool) -> None:
        """Update mastery tracking based on answer correctness"""
        if session_id not in self._mastery_tracking:
            return
        
        tracking = self._mastery_tracking[session_id]
        user_id = tracking.get("user_id")
        current_strategy = tracking.get("current_strategy")
        topic = tracking.get("topic", "")
        
        # Update attempts and correctness
        tracking["attempts"] = tracking.get("attempts", 0) + 1
        if is_correct:
            tracking["correct_answers"] = tracking.get("correct_answers", 0) + 1
            tracking["current_streak"] = tracking.get("current_streak", 0) + 1
            tracking["retry_count"] = 0
        else:
            tracking["current_streak"] = 0
            tracking["retry_count"] = tracking.get("retry_count", 0) + 1
        
        # Update mastery level (0-10 scale)
        if is_correct:
            # Increase mastery level
            tracking["mastery_level"] = min(10, tracking.get("mastery_level", 0) + 0.5)
        else:
            # Decrease mastery level on incorrect answers
            tracking["mastery_level"] = max(0, tracking.get("mastery_level", 0) - 0.2)
        
        # Update difficulty based on mastery level
        mastery_level = tracking.get("mastery_level", 0)
        if mastery_level >= 8:
            tracking["current_difficulty"] = "expert"
        elif mastery_level >= 5:
            tracking["current_difficulty"] = "advanced"
        elif mastery_level >= 2:
            tracking["current_difficulty"] = "intermediate"
        else:
            tracking["current_difficulty"] = "beginner"
        
        # Consider strategy change if struggling
        retry_count = tracking.get("retry_count", 0)
        last_change = tracking.get("last_strategy_change", datetime.now())
        strategy_cooldown = timedelta(minutes=5)  # Don't change strategies too frequently
        
        if retry_count >= 3 and datetime.now() - last_change > strategy_cooldown:
            # Try a different strategy
            current = tracking.get("current_strategy")
            available = [s for s in self.STRATEGIES if s != current]
            
            # If we have performance data, choose the best performing alternative
            if user_id in self._strategy_performance:
                user_strats = self._strategy_performance[user_id]
                available_with_scores = [
                    (s, user_strats.get(s, {}).get("success_rate", 0.5)) 
                    for s in available
                ]
                new_strategy = max(available_with_scores, key=lambda x: x[1])[0]
            else:
                # Otherwise randomly select
                new_strategy = random.choice(available)
            
            # Log strategy change to audit log
            try:
                from agent.utils.audit import log_strategy_change
                log_strategy_change(
                    session_id=session_id,
                    user_id=user_id,
                    old_strategy=current,
                    new_strategy=new_strategy,
                    reason=f"Retry count {retry_count} exceeded threshold",
                    topic=topic,
                    retry_count=retry_count
                )
            except ImportError:
                # Audit module not available, continue without logging
                pass
            
            tracking["current_strategy"] = new_strategy
            tracking["last_strategy_change"] = datetime.now()
            tracking["retry_count"] = 0
        
        # Update strategy performance metrics
        if user_id:
            if user_id not in self._strategy_performance:
                self._strategy_performance[user_id] = {}
            
            if current_strategy not in self._strategy_performance[user_id]:
                self._strategy_performance[user_id][current_strategy] = {
                    "attempts": 0,
                    "correct": 0,
                    "success_rate": 0.0
                }
            
            strat_stats = self._strategy_performance[user_id][current_strategy]
            old_success_rate = strat_stats.get("success_rate", 0.0)
            
            strat_stats["attempts"] = strat_stats.get("attempts", 0) + 1
            if is_correct:
                strat_stats["correct"] = strat_stats.get("correct", 0) + 1
            
            # Update success rate
            attempts = strat_stats.get("attempts", 0)
            correct = strat_stats.get("correct", 0)
            new_success_rate = correct / max(attempts, 1)
            strat_stats["success_rate"] = new_success_rate
            
            # Log memory rule update to audit log
            try:
                from agent.utils.audit import log_memory_rule_update
                rule_id = f"{user_id}:{current_strategy}"
                log_memory_rule_update(
                    rule_id=rule_id,
                    user_id=user_id,
                    strategy=current_strategy,
                    old_success_rate=old_success_rate,
                    new_success_rate=new_success_rate
                )
            except ImportError:
                # Audit module not available, continue without logging
                pass