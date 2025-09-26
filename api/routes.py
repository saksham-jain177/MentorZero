import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from agent.db.base import get_session_factory
from agent.llm.ollama_client import OllamaClient
from agent.services.teaching import TeachingService
from agent.services.validators import AZLValidators
from agent.azl.generator import AZLGenerator
from agent.azl.scorer import AZLScorer
from agent.config import get_azl_config
from agent.voice.stt import WhisperSTT, STT_AVAILABLE
from agent.voice.tts import ChatterboxTTS, TTS_AVAILABLE
from api.deps import get_ollama_client, get_teaching_service, get_db_session, get_judge_ollama_client, background_processor
from agent.config import get_settings

VOICE_AVAILABLE = STT_AVAILABLE and TTS_AVAILABLE

router = APIRouter()

class TeachRequest(BaseModel):
    userId: str
    topic: str
    mode: str = "explain"
    sessionId: Optional[str] = None

class TeachResponse(BaseModel):
    session_id: str
    first: str

class SubmitAnswerRequest(BaseModel):
    sessionId: str
    userAnswer: str

class SubmitAnswerResponse(BaseModel):
    explanation: str
    next_item: str

class OptimizeRequest(BaseModel):
    sessionId: Optional[str] = None
    text: str

class OptimizeResponse(BaseModel):
    optimized: str

class ReaskRequest(BaseModel):
    topic: str
    variant: str = "simplify"  # simplify | expand | add_examples
    sessionId: Optional[str] = None
    userId: Optional[str] = None

class ReaskResponse(BaseModel):
    text: str

class UploadTextRequest(BaseModel):
    sessionId: Optional[str] = None
    text: str
    url: Optional[str] = None
    topK: Optional[int] = None
    mmr: Optional[float] = None

class ScrapeRequest(BaseModel):
    url: str
    sessionId: Optional[str] = None

class ScrapeResponse(BaseModel):
    session_id: str
    extracted_chunks_count: int

class UploadTextResponse(BaseModel):
    extracted_chunks_count: int
    session_id: Optional[str] = None

class ProgressResponse(BaseModel):
    accuracy: float
    streak: int
    topics_completed: int
    topics_in_progress: List[Dict[str, Any]]
    mastery_level: int = 0
    current_difficulty: str = "beginner"
    current_strategy: str = "neural_compression"
    attempts: int = 0
    correct_answers: int = 0

class RollbackRequest(BaseModel):
    rule_id: str
    user_id: str
    reason: str
    target_timestamp: Optional[str] = None

class RollbackResponse(BaseModel):
    success: bool
    message: str
    old_success_rate: Optional[float] = None
    new_success_rate: Optional[float] = None

class LLMHealthResponse(BaseModel):
    host: str
    model: str
    ready: bool

class VoiceHealthResponse(BaseModel):
    stt_available: bool
    tts_available: bool

class TimingsResponse(BaseModel):
    ingest_ms: Optional[float] = None
    retrieve_ms: Optional[float] = None
    judge_ms: Optional[float] = None

class STTRequest(BaseModel):
    language: str = "en"

class TTSRequest(BaseModel):
    text: str
    voice: str = "default"

class AZLProposalRequest(BaseModel):
    topic: str
    count: int = 5

class AZLProposalResponse(BaseModel):
    examples: List[Dict[str, str]]
    proposal_id: str

class AZLValidationRequest(BaseModel):
    proposal_id: str
    example_idx: int

class AZLValidationResponse(BaseModel):
    validation_results: Dict[str, Any]
    passed: bool

class AZLAcceptRequest(BaseModel):
    proposal_id: str
    example_idx: int
    accepted: bool
    message: str

class AZLRegenerateRequest(BaseModel):
    proposal_id: str
    example_idx: int
    topic: Optional[str] = None
    style_hint: Optional[str] = None

class AZLScoreRequest(BaseModel):
    question: str
    answer: str

class AZLScoreResponse(BaseModel):
    checks: Dict[str, Any]
    judge: Dict[str, Any]
    score: float
    passed: bool

class AZLAutoLearnRequest(BaseModel):
    topic: str
    count: int = 10
    pass_threshold: Optional[float] = None
    max_attempts: Optional[int] = None

class AZLAutoLearnItem(BaseModel):
    example: Dict[str, Any]
    score: float
    passed: bool
    attempts: int
    checks: Dict[str, Any]
    judge: Dict[str, Any]

class AZLAutoLearnResponse(BaseModel):
    accepted_count: int
    failed_count: int
    items: List[AZLAutoLearnItem]

def _sse_format(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")

@router.get("/llm_health", response_model=LLMHealthResponse)
async def check_llm_health(
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Check if the LLM is available and ready."""
    try:
        await ollama_client.health_check()
        return {
            "host": ollama_client.base_url,
            "model": ollama_client.model,
            "ready": True
        }
    except Exception as e:
        return {
            "host": ollama_client.base_url,
            "model": ollama_client.model,
            "ready": False
        }

@router.get("/voice_health", response_model=VoiceHealthResponse)
async def check_voice_health():
    """Check if voice services are available."""
    if not VOICE_AVAILABLE:
        return {
            "stt_available": False,
            "tts_available": False
        }
    
    return {
        "stt_available": STT_AVAILABLE,
        "tts_available": TTS_AVAILABLE
    }

@router.get("/judge_health", response_model=LLMHealthResponse)
async def check_judge_health():
    """Check if the Judge LLM is available and ready."""
    try:
        from api.deps import get_judge_ollama_client
        judge = get_judge_ollama_client()
        await judge.health_check()
        return {
            "host": judge.base_url,
            "model": getattr(judge, "model", ""),
            "ready": True
        }
    except Exception:
        try:
            judge = get_judge_ollama_client()
            return {
                "host": judge.base_url,
                "model": getattr(judge, "model", ""),
                "ready": False
            }
        except Exception:
            return {"host": "", "model": "", "ready": False}

async def classify_query_intent(query: str, ollama_client: OllamaClient) -> Dict[str, Any]:
    """Classify if query is a topic request or conversational question."""
    try:
        prompt = f"""Analyze this user query and determine the intent:

Query: "{query}"

Classify as either:
1. TOPIC_REQUEST - User wants structured learning on a specific topic (e.g., "Machine Learning", "Python programming")
2. CONVERSATION - User is asking a conversational question (e.g., "Can you tell me about AI?", "What is quantum computing?")

If CONVERSATION, extract the core topic they're asking about.

Respond in JSON format:
{{
    "intent": "TOPIC_REQUEST" or "CONVERSATION",
    "extracted_topic": "core topic if CONVERSATION, otherwise same as input",
    "confidence": 0.0-1.0
}}"""
        
        response = await ollama_client.send(prompt=prompt, temperature=0.1)
        result_text = (response or {}).get("text", "").strip()
        
        # Parse JSON response
        import json
        try:
            result = json.loads(result_text)
            return {
                "intent": result.get("intent", "TOPIC_REQUEST"),
                "extracted_topic": result.get("extracted_topic", query),
                "confidence": float(result.get("confidence", 0.8))
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback: simple keyword detection
            question_words = ["what", "how", "can", "could", "would", "tell", "explain", "describe"]
            is_question = any(word in query.lower() for word in question_words)
            return {
                "intent": "CONVERSATION" if is_question else "TOPIC_REQUEST",
                "extracted_topic": query,
                "confidence": 0.6
            }
    except Exception as e:
        logger.warning(f"Query classification failed: {e}")
        return {
            "intent": "TOPIC_REQUEST",
            "extracted_topic": query,
            "confidence": 0.5
        }

async def generate_conversational_response(
    ollama_client: OllamaClient, 
    original_query: str, 
    extracted_topic: str, 
    context: str
) -> str:
    """Generate a natural conversational response to user's question."""
    try:
        # Build context-aware prompt for conversational response
        context_section = f"\n\nRelevant information:\n{context}" if context.strip() else ""
        
        prompt = f"""You are a helpful AI tutor. The user asked: "{original_query}"

Provide a natural, conversational response that directly answers their question about {extracted_topic}. Be engaging and educational, but keep it conversational rather than formal.{context_section}

Guidelines:
- Answer their specific question directly
- Use a friendly, conversational tone
- Include relevant details from the context if available
- Keep it concise but informative
- End with an invitation to ask follow-up questions

Response:"""
        
        response = await ollama_client.send(prompt=prompt, temperature=0.7)
        result_text = (response or {}).get("text", "").strip()
        
        if not result_text:
            # Fallback response
            return f"I'd be happy to help you learn about {extracted_topic}! Could you be more specific about what aspect you'd like to explore?"
        
        return result_text
        
    except Exception as e:
        logger.warning(f"Conversational response generation failed: {e}")
        return f"I'd be happy to help you learn about {extracted_topic}! Could you be more specific about what aspect you'd like to explore?"

@router.post("/teach", response_model=TeachResponse)
async def teach(
    request: TeachRequest,
    teaching_service: TeachingService = Depends(get_teaching_service),
    ollama_client: OllamaClient = Depends(get_ollama_client),
    db: Session = Depends(get_db_session),
):
    """Start a teaching session with natural language support."""
    try:
        # Validate required fields
        if not request.userId or not request.topic:
            logger.error(f"Missing required fields: userId='{request.userId}', topic='{request.topic}'")
            raise HTTPException(status_code=422, detail="Missing required fields: userId and topic are required")
        
        # Classify query intent
        classification = await classify_query_intent(request.topic, ollama_client)
        logger.info(f"Query classification: {classification}")
        
        # Use extracted topic for structured learning
        sanitized_topic = classification["extracted_topic"].strip()
        sanitized_mode = (request.mode or "explain").strip().lower()
        if sanitized_mode not in ("explain", "quiz"):
            sanitized_mode = "explain"
        logger.info(f"Teach sanitize: topic='{sanitized_topic}', mode='{sanitized_mode}', intent={classification['intent']}")
        # Always create a new session ID for fresh requests
        if not request.sessionId:
            session_id = str(uuid4())
        else:
            session_id = request.sessionId
            
        # Always clear existing session data for this topic to prevent caching
        # This ensures we always get a fresh response for each query
        teaching_service._sessions = {}
        teaching_service._contexts = {}
        teaching_service._mastery_tracking = {}
        logger.info(f"Cleared all existing sessions for fresh request on topic: {request.topic}")
        
        # Initialize mastery tracking for Loop-Until-Mastered
        teaching_service._initialize_mastery_tracking(session_id, request.userId, sanitized_topic)
        
        # Create user and session in database
        from agent.db.crud import create_user, create_session, get_user
        
        # Create or get user
        user = get_user(db, request.userId)
        if not user:
            user = create_user(db, request.userId)
        
        # Create session
        create_session(db, session_id, request.userId, sanitized_topic)
        db.commit()
        
        # Retrieve Agentic RAG context (required)
        context = await teaching_service.retrieve_context(session_id, sanitized_topic)
        
        # Log the request details
        logger.info(f"Teaching request: user={request.userId}, topic={sanitized_topic}, mode={sanitized_mode}, session_id={session_id}")
        
        # Generate response based on intent
        try:
            if classification["intent"] == "CONVERSATION":
                # Generate conversational response
                first_item = await generate_conversational_response(
                    ollama_client, 
                    request.topic,  # Use original query for context
                    sanitized_topic,  # Use extracted topic for RAG
                    context
                )
            else:
                # Generate structured teaching item
                first_item = await teaching_service.generate_explanation(
                    ollama_client, 
                    request.userId, 
                    sanitized_topic, 
                    sanitized_mode,
                    session_id,
                    context
                )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to a simple response
            first_item = f"I'd be happy to help you learn about {sanitized_topic}! Could you be more specific about what aspect you'd like to explore?"
        
        # Log the response
        logger.info(f"Generated explanation for session {session_id}, topic: {sanitized_topic}")

        # Add explicit cache-control to avoid any intermediate caching
        from fastapi import Response
        import json as _json
        return Response(
            content=_json.dumps({
                "session_id": session_id,
                "first": first_item
            }),
            media_type="application/json",
            headers={"Cache-Control": "no-store"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start teaching: {str(e)}")

@router.post("/submit_answer", response_model=SubmitAnswerResponse)
async def submit_answer(
    request: SubmitAnswerRequest,
    teaching_service: TeachingService = Depends(get_teaching_service),
    ollama_client: OllamaClient = Depends(get_ollama_client),
    db: Session = Depends(get_db_session),
):
    """Submit an answer and get feedback."""
    try:
        # Get session info
        session_info = teaching_service.get_session_info(request.sessionId)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        user_id = session_info.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="Session has no user ID")
        
        # Retrieve Agentic RAG context (required)
        context = await teaching_service.retrieve_context(request.sessionId, session_info.get("topic", ""))
        
        # Get current strategy from teaching service
        tracking = teaching_service._mastery_tracking.get(request.sessionId, {})
        current_strategy = tracking.get("current_strategy", "neural_compression")
        
        # Generate feedback with Loop-Until-Mastered tracking
        explanation, is_correct = await teaching_service.generate_feedback(
            ollama_client,
            request.sessionId,
            request.userAnswer,
            context
        )
        
        # Generate next item based on current mastery level and strategy
        next_item = await teaching_service.generate_next_item(
            ollama_client,
            request.sessionId,
            context
        )
        
        # Record interaction in database
        from uuid import uuid4
        from datetime import datetime, timezone
        from agent.db.crud import add_interaction, get_session, upsert_memory_rule, get_user_memory_rules
        
        # Get turn index
        db_session = get_session(db, request.sessionId)
        if db_session:
            # Count existing interactions
            turn_index = len(db_session.interactions) if hasattr(db_session, 'interactions') else 0
            
            # Add the interaction
            interaction_id = str(uuid4())
            add_interaction(
                db=db,
                interaction_id=interaction_id,
                session_id=request.sessionId,
                turn_index=turn_index,
                input_text=request.userAnswer,
                agent_response=explanation,
                agent_strategy_tag=current_strategy,
                outcome="correct" if is_correct else "incorrect",
                confidence_score=None  # Could add confidence in future
            )
            
            # Update memory rule for this strategy
            rule_id = f"{user_id}:{current_strategy}"
            
            # Get existing rules for this user
            existing_rules = get_user_memory_rules(db, user_id)
            existing_rule = next((r for r in existing_rules if r.rule_id == rule_id), None)
            
            # Calculate new success rate
            attempts = 1
            correct = 1 if is_correct else 0
            success_rate = 0.0
            
            if existing_rule:
                # Extract previous stats from rule strategy field if it's JSON
                try:
                    import json
                    stats = json.loads(existing_rule.strategy)
                    if isinstance(stats, dict) and "attempts" in stats:
                        attempts += stats.get("attempts", 0)
                        correct += stats.get("correct", 0)
                except:
                    # If not JSON, just use the success rate directly
                    attempts = 10  # Assume some history
                    correct = int(existing_rule.success_rate * attempts)
                    correct += 1 if is_correct else 0
                    attempts += 1
            
            success_rate = correct / attempts if attempts > 0 else 0.0
            
            # Store stats in strategy field as JSON
            strategy_stats = {
                "name": current_strategy,
                "attempts": attempts,
                "correct": correct,
                "description": get_strategy_description(current_strategy)
            }
            
            # Update the memory rule
            upsert_memory_rule(
                db=db,
                rule_id=rule_id,
                strategy=json.dumps(strategy_stats),
                user_id=user_id,
                success_rate=success_rate,
                last_used_ts=datetime.now(timezone.utc)
            )
            
            # Commit the transaction
            db.commit()
        
        from fastapi import Response
        import json as _json
        return Response(
            content=_json.dumps({
                "explanation": explanation,
                "next_item": next_item
            }),
            media_type="application/json",
            headers={"Cache-Control": "no-store"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_text(
    request: OptimizeRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Reformat/optimize an LLM answer for structure and clarity."""
    try:
        system_prompt = (
            "You are a formatting optimizer. Improve structure and clarity of the provided content without changing facts. "
            "Rules: Use markdown only; keep headings, bullets, and numbered lists consistent; ensure blank lines between sections; "
            "wrap lines naturally; do not add self-referential text; do not change meaning."
        )
        prompt = (
            "Rewrite and format the following content into clean markdown with consistent headings, bullet points (•) or '-', and numbered lists (1., 2., 3.).\n\n" 
            f"CONTENT:\n{request.text}"
        )
        resp = await ollama_client.send(prompt=prompt, system_prompt=system_prompt, temperature=0.0)
        text = resp.get("text", "")
        from fastapi import Response
        import json as _json
        return Response(content=_json.dumps({"optimized": text}), media_type="application/json", headers={"Cache-Control": "no-store"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize: {str(e)}")


@router.post("/reask", response_model=ReaskResponse)
async def reask_variant(
    request: ReaskRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Re-ask the last user query with a different guidance (simplify, expand, add examples)."""
    try:
        variant = (request.variant or "simplify").lower()
        if variant == "expand":
            sys = (
                "You are MentorZero. Re-answer the topic with deeper coverage and additional context. "
                "Use clear headings, subpoints, and multiple real-world examples. Keep it precise and educational."
            )
        elif variant == "add_examples":
            sys = (
                "You are MentorZero. Re-answer the topic with a focus on concrete, varied examples and short exercises. "
                "Keep sections concise, add bullet lists and small coding or thought exercises where relevant."
            )
        else:
            sys = (
                "You are MentorZero. Re-answer the topic in a simpler, more concise manner without losing key facts. "
                "Prefer short sentences, bullets, and compact structure."
            )

        prompt = (
            f"Topic: {request.topic}. Provide a well-structured answer with consistent markdown formatting."
        )
        resp = await ollama_client.send(prompt=prompt, system_prompt=sys, temperature=0.0)
        text = resp.get("text", "")
        from fastapi import Response
        import json as _json
        return Response(content=_json.dumps({"text": text}), media_type="application/json", headers={"Cache-Control": "no-store"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-ask: {str(e)}")


def get_strategy_description(strategy_name: str) -> str:
    """Get a human-readable description of a teaching strategy"""
    descriptions = {
        "neural_compression": "Uses memory hooks and compression techniques to enhance recall",
        "socratic_dialogue": "Uses questions to guide the learner to insights",
        "concrete_examples": "Uses real-world examples to illustrate concepts",
        "visual_mapping": "Uses visual metaphors and diagrams to organize knowledge",
        "spaced_repetition": "Revisits concepts at optimal intervals to improve retention"
    }
    return descriptions.get(strategy_name, "Custom teaching strategy")

@router.post("/memory_rule/rollback", response_model=RollbackResponse)
async def rollback_memory_rule(
    request: RollbackRequest,
    db: Session = Depends(get_db_session),
):
    """Roll back a memory rule to a previous state."""
    try:
        # Import audit utilities
        from agent.utils.audit import rollback_memory_rule as audit_rollback
        from agent.utils.audit import get_memory_rule_history
        from agent.db.crud import get_memory_rule, upsert_memory_rule
        
        # Get the current rule
        current_rule = get_memory_rule(db, request.rule_id)
        if not current_rule:
            return {
                "success": False,
                "message": "Memory rule not found"
            }
        
        # Get the target state to roll back to
        target_state = audit_rollback(
            rule_id=request.rule_id,
            user_id=request.user_id,
            reason=request.reason,
            target_timestamp=request.target_timestamp
        )
        
        if not target_state:
            return {
                "success": False,
                "message": "No previous state found to roll back to"
            }
        
        # Get the old success rate
        old_success_rate = current_rule.success_rate
        
        # Update the rule with the target state
        new_success_rate = target_state.get("new_success_rate", 0.0)
        
        # Update the rule in the database
        upsert_memory_rule(
            db=db,
            rule_id=request.rule_id,
            strategy=current_rule.strategy,  # Keep the strategy content
            user_id=current_rule.user_id,
            success_rate=new_success_rate,
            last_used_ts=datetime.now(timezone.utc)
        )
        
        # Commit the transaction
        db.commit()
        
        return {
            "success": True,
            "message": f"Memory rule rolled back to {target_state.get('timestamp')}",
            "old_success_rate": old_success_rate,
            "new_success_rate": new_success_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to roll back memory rule: {str(e)}")

@router.post("/upload_url_or_text", response_model=UploadTextResponse)
async def upload_text(
    request: UploadTextRequest,
    teaching_service: TeachingService = Depends(get_teaching_service),
    db: Session = Depends(get_db_session),
):
    """Upload text for RAG."""
    try:
        t0 = datetime.now()
        # Ensure a session id exists for indexing namespace
        from uuid import uuid4 as _uuid4
        session_id = request.sessionId or str(_uuid4())
        # Process the text
        chunks_count = teaching_service.process_text(session_id, request.text)
        dt = (datetime.now() - t0).total_seconds() * 1000.0
        _mark_timing("ingest_ms", dt)
        return {
            "extracted_chunks_count": chunks_count,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

@router.post("/scrape_fetch", response_model=ScrapeResponse)
async def scrape_fetch(
    request: ScrapeRequest,
    teaching_service: TeachingService = Depends(get_teaching_service),
):
    """Fetch a URL, extract readable text, and ingest into RAG."""
    try:
        from uuid import uuid4 as _uuid4
        import re
        import httpx
        session_id = request.sessionId or str(_uuid4())
        url = (request.url or '').strip()
        if not url:
            raise HTTPException(status_code=400, detail="url is required")
        # Simple fetch
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch: {resp.status_code}")
            html = resp.text
        # Very basic readability: strip tags
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No readable text extracted")
        count = teaching_service.process_text(session_id, text)
        return {"session_id": session_id, "extracted_chunks_count": count}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape: {str(e)}")

@router.get("/progress/{session_id}", response_model=ProgressResponse)
async def get_progress(
    session_id: str,
    teaching_service: TeachingService = Depends(get_teaching_service),
    db: Session = Depends(get_db_session),
):
    """Get the progress for a session."""
    try:
        t0 = datetime.now()
        # First try to get progress from database
        from agent.db.crud import get_session_progress_metrics
        
        # Get database metrics
        db_metrics = get_session_progress_metrics(db, session_id)
        
        # If we have a session in the database, use those metrics
        if db_metrics and not db_metrics.get("error"):
            return db_metrics
        
        # Fall back to in-memory metrics from teaching service
        progress = teaching_service.get_progress(session_id)
        dt = (datetime.now() - t0).total_seconds() * 1000.0
        _mark_timing("retrieve_ms", dt)
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

# Simple in-memory timings (best-effort)
_timings: Dict[str, float] = {}

def _mark_timing(key: str, ms: float) -> None:
    try:
        _timings[key] = float(ms)
    except Exception:
        pass

@router.get("/metrics/timings", response_model=TimingsResponse)
async def timings_metrics():
    try:
        return {
            "ingest_ms": _timings.get("ingest_ms"),
            "retrieve_ms": _timings.get("retrieve_ms"),
            "judge_ms": _timings.get("judge_ms"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get timings: {str(e)}")

@router.post("/azl/submit_background")
async def submit_azl_background(
    request: AZLAutoLearnRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Submit AZL evaluation to background processing queue."""
    try:
        # Register handler if not already registered
        if 'azl_evaluation' not in background_processor._task_handlers:
            async def azl_handler(payload, progress_callback):
                # Import here to avoid circular dependencies
                from agent.azl.optimized_scorer import OptimizedAZLScorer
                
                cfg = get_azl_config()
                judge_client = get_judge_ollama_client()
                db_session = get_db_session()
                
                scorer = OptimizedAZLScorer(
                    ollama_client,
                    cfg.score_weights,
                    judge_client,
                    db_session=db_session,
                    cache_ttl_seconds=3600,
                    confidence_threshold=0.7,
                    score_margin=0.1
                )
                
                generator = AZLGenerator(ollama_client)
                examples, proposal_id = await generator.propose_examples(
                    payload['topic'], 
                    payload['count']
                )
                
                results = []
                total = len(examples)
                
                for idx, ex in enumerate(examples):
                    # Update progress
                    progress = (idx / total) * 100
                    progress_callback(progress)
                    
                    result = await scorer.score(ex, payload['topic'])
                    results.append({
                        'example': ex,
                        'score': result['score'],
                        'passed': result['passed'],
                        'checks': result['checks'],
                        'judge': result.get('judge', {}),
                        'method': result.get('method', 'unknown')
                    })
                
                progress_callback(100)
                
                return {
                    'proposal_id': proposal_id,
                    'results': results,
                    'accepted_count': sum(1 for r in results if r['passed']),
                    'failed_count': sum(1 for r in results if not r['passed'])
                }
            
            background_processor.register_handler('azl_evaluation', azl_handler)
        
        # Submit task
        task_id = await background_processor.submit_task(
            'azl_evaluation',
            {
                'topic': request.topic,
                'count': request.count,
                'pass_threshold': request.pass_threshold,
                'max_attempts': request.max_attempts
            }
        )
        
        return {
            'task_id': task_id,
            'status': 'submitted',
            'message': f'AZL evaluation for "{request.topic}" submitted to background queue'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit background task: {str(e)}")

@router.get("/azl/task_status/{task_id}")
async def get_azl_task_status(task_id: str):
    """Get the status of a background AZL task."""
    task = await background_processor.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return {
        'task_id': task.task_id,
        'status': task.status,
        'progress': task.progress,
        'created_at': task.created_at,
        'started_at': task.started_at,
        'completed_at': task.completed_at,
        'result': task.result if task.status == 'completed' else None,
        'error': task.error if task.status == 'failed' else None
    }

@router.get("/metrics/performance")
async def get_performance_metrics(db: Session = Depends(get_db_session)):
    """Get comprehensive performance metrics."""
    try:
        from agent.db.models import PerformanceMetrics
        from sqlalchemy import func
        
        # Get recent metrics (last 24 hours)
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        
        metrics = db.query(PerformanceMetrics).filter(
            PerformanceMetrics.created_at >= since
        ).all()
        
        # Aggregate by operation
        operation_stats = {}
        for metric in metrics:
            op = metric.operation
            if op not in operation_stats:
                operation_stats[op] = {
                    "count": 0,
                    "total_duration_ms": 0.0,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_duration_ms": 0.0,
                    "min_duration_ms": float('inf'),
                    "max_duration_ms": 0.0
                }
            
            stats = operation_stats[op]
            stats["count"] += 1
            stats["total_duration_ms"] += metric.duration_ms
            if metric.success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
            stats["min_duration_ms"] = min(stats["min_duration_ms"], metric.duration_ms)
            stats["max_duration_ms"] = max(stats["max_duration_ms"], metric.duration_ms)
        
        # Calculate averages
        for stats in operation_stats.values():
            if stats["count"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["success_rate"] = stats["success_count"] / stats["count"]
            if stats["min_duration_ms"] == float('inf'):
                stats["min_duration_ms"] = 0.0
        
        return {
            "period_hours": 24,
            "total_operations": len(metrics),
            "operations": operation_stats,
            "cache_performance": {
                "memory_hits": sum(1 for m in metrics if m.operation == "cache_hit_memory"),
                "db_hits": sum(1 for m in metrics if m.operation == "cache_hit_db"), 
                "misses": sum(1 for m in metrics if m.operation == "cache_miss")
            }
        }
        
    except Exception as e:
        return {"error": f"Failed to get metrics: {str(e)}", "operations": {}}

@router.post("/stt", response_model=Dict[str, str])
async def speech_to_text(
    language: str = "en",
    audio_file: UploadFile = File(...),
):
    """Convert speech to text."""
    if not STT_AVAILABLE:
        raise HTTPException(status_code=501, detail="Speech-to-text is not available")
    
    try:
        stt = WhisperSTT()
        content = await audio_file.read()
        
        # Process the audio
        text = await stt.transcribe(content, language)
        
        return {
            "text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

@router.post("/tts", response_model=Dict[str, str])
async def text_to_speech(
    request: TTSRequest,
):
    """Convert text to speech."""
    if not TTS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Text-to-speech is not available")
    
    try:
        tts = ChatterboxTTS()
        
        # Generate audio
        audio_path = await tts.generate(request.text, request.voice)
        
        return {
            "audio_path": audio_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

@router.post("/azl/propose", response_model=AZLProposalResponse)
async def propose_examples(
    request: AZLProposalRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Propose synthetic examples for a topic."""
    try:
        generator = AZLGenerator(ollama_client)
        examples, proposal_id = await generator.propose_examples(request.topic, request.count)
        
        # Log the proposal to audit log
        try:
            from agent.utils.audit import log_azl_proposal
            log_azl_proposal(
                proposal_id=proposal_id,
                topic=request.topic,
                user_id=None,  # No user ID available in this context
                example_count=len(examples)
            )
        except ImportError:
            # Audit module not available, continue without logging
            pass
        
        return {
            "examples": examples,
            "proposal_id": proposal_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate examples: {str(e)}")

@router.post("/azl/validate", response_model=AZLValidationResponse)
async def validate_example(
    request: AZLValidationRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Validate a synthetic example."""
    try:
        validators = AZLValidators(ollama_client)
        example = AZLGenerator.get_example(request.proposal_id, request.example_idx)
        
        if not example:
            raise HTTPException(status_code=404, detail="Example not found")
        
        results = await validators.validate_all(example)
        passed = all(result.get("passed", False) for result in results.values())
        
        # Log the validation to audit log
        try:
            from agent.utils.audit import log_azl_validation
            log_azl_validation(
                proposal_id=request.proposal_id,
                example_idx=request.example_idx,
                validation_results=results,
                passed=passed
            )
        except ImportError:
            # Audit module not available, continue without logging
            pass
        
        return {
            "validation_results": results,
            "passed": passed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate example: {str(e)}")

@router.post("/azl/accept", response_model=Dict[str, bool])
async def accept_example(
    request: AZLAcceptRequest,
    teaching_service: TeachingService = Depends(get_teaching_service),
    db: Session = Depends(get_db_session),
):
    """Accept or reject a synthetic example."""
    try:
        example = AZLGenerator.get_example(request.proposal_id, request.example_idx)
        
        if not example:
            raise HTTPException(status_code=404, detail="Example not found")
        
        # Log the approval/rejection to audit log
        try:
            from agent.utils.audit import log_azl_approval
            log_azl_approval(
                proposal_id=request.proposal_id,
                example_idx=request.example_idx,
                user_id="system",  # In a real app, this would be the actual user ID
                accepted=request.accepted,
                message=request.message
            )
        except ImportError:
            # Audit module not available, continue without logging
            pass
        
        if request.accepted:
            teaching_service.store_synthetic_example(example, request.message)
        
        return {
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process example: {str(e)}")

@router.post("/azl/regenerate", response_model=Dict[str, Any])
async def regenerate_example(
    request: AZLRegenerateRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Regenerate a single AZL example and update the proposal store."""
    try:
        # Get current example for context
        current = AZLGenerator.get_example(request.proposal_id, request.example_idx)
        if current is None:
            raise HTTPException(status_code=404, detail="Example not found")

        topic = (request.topic or current.get("question", "")).strip()
        style_hint = (request.style_hint or "").strip()

        guidance = (
            "Generate 1 improved question-answer pair for the topic. Keep it precise, educational, and non-redundant."
        )
        if style_hint:
            guidance += f" Style hint: {style_hint}."

        prompt = (
            f"Topic: {topic or 'general'}\n"
            f"Current QA to improve:\nQ: {current.get('question','')}\nA: {current.get('answer','')}\n\n"
            f"{guidance}\n"
            "Return JSON of shape {\"question\":\"...\", \"answer\":\"...\"}."
        )

        resp = await ollama_client.send(prompt=prompt, temperature=0.0)
        text = resp.get("text", "")

        # Use robust parser from generator
        new_example = AZLGenerator.parse_single_example(text)
        if not new_example:
            raise HTTPException(status_code=500, detail="Failed to parse regenerated example")

        # Update store
        updated = AZLGenerator.set_example(request.proposal_id, request.example_idx, new_example)
        if not updated:
            raise HTTPException(status_code=404, detail="Proposal not found")

        from fastapi import Response
        import json as _json
        return Response(content=_json.dumps({"example": new_example}), media_type="application/json", headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to regenerate example: {str(e)}")

@router.post("/azl/score", response_model=AZLScoreResponse)
async def score_example(
    request: AZLScoreRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Score a single Q/A using combined validators and LLM judge."""
    try:
        cfg = get_azl_config()
        judge_client = get_judge_ollama_client()
        scorer = AZLScorer(ollama_client, cfg.score_weights, judge_client)
        result = await scorer.score({"question": request.question, "answer": request.answer})
        # apply threshold to determine pass
        passed = result["score"] >= cfg.pass_threshold
        return {"checks": result["checks"], "judge": result["judge"], "score": result["score"], "passed": passed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to score example: {str(e)}")

@router.post("/azl/autolearn", response_model=AZLAutoLearnResponse)
async def azl_autolearn(
    request: AZLAutoLearnRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
    teaching_service: TeachingService = Depends(get_teaching_service),
):
    """Run propose→self-judge→autoregenerate until pass or max_attempts for a batch."""
    try:
        cfg = get_azl_config()
        pass_threshold = request.pass_threshold if request.pass_threshold is not None else cfg.pass_threshold
        max_attempts = request.max_attempts if request.max_attempts is not None else cfg.max_attempts

        generator = AZLGenerator(ollama_client)
        examples, proposal_id = await generator.propose_examples(request.topic, request.count)
        judge_client = get_judge_ollama_client()
        scorer = AZLScorer(ollama_client, cfg.score_weights, judge_client)

        results: List[AZLAutoLearnItem] = []
        accepted_count = 0

        for idx, ex in enumerate(examples):
            attempts = 0
            best = None
            while attempts <= max_attempts:
                attempts += 1
                r = await scorer.score(ex)
                if r["score"] >= pass_threshold:
                    accepted_count += 1
                    # store in KB/examples
                    teaching_service.store_synthetic_example(ex, f"auto-accepted score={r['score']:.2f}")
                    results.append(AZLAutoLearnItem(example=ex, score=r["score"], passed=True, attempts=attempts, checks=r["checks"], judge=r["judge"]))
                    break
                best = r
                # regenerate
                regen_req = AZLRegenerateRequest(proposal_id=proposal_id, example_idx=idx, topic=request.topic)
                # inline regenerate logic to avoid round-trip
                current = AZLGenerator.get_example(proposal_id, idx)
                if not current:
                    break
                guidance = "Generate 1 improved question-answer pair for the topic. Keep it precise and factual. Return JSON {\"question\":..., \"answer\":...}."
                prompt = (
                    f"Topic: {request.topic or 'general'}\n"
                    f"Current QA to improve:\nQ: {current.get('question','')}\nA: {current.get('answer','')}\n\n{guidance}"
                )
                regen_resp = await ollama_client.send(prompt=prompt, temperature=0.0)
                import json as _json, re as _re
                payload = regen_resp.get("text", "")
                m = _re.search(r"```json\s*([\s\S]*?)\s*```", payload)
                obj_txt = m.group(1) if m else payload
                try:
                    new_ex = _json.loads(obj_txt)
                except Exception:
                    m2 = _re.search(r"\{[\s\S]*?\}", payload)
                    if not m2:
                        break
                    new_ex = _json.loads(m2.group(0))
                if isinstance(new_ex, dict) and "question" in new_ex and "answer" in new_ex:
                    AZLGenerator.set_example(proposal_id, idx, new_ex)
                    ex = new_ex
                else:
                    break
            else:
                pass

            if attempts == 0 or (best and best["score"] < pass_threshold and (len(results) == 0 or results[-1].example is not ex)):
                # failed case
                results.append(AZLAutoLearnItem(example=ex, score=(best or {"score": 0.0})["score"], passed=False, attempts=attempts, checks=(best or {"checks": {}})["checks"], judge=(best or {"judge": {}})["judge"]))

        failed_count = len([it for it in results if not it.passed])
        return {
            "accepted_count": accepted_count,
            "failed_count": failed_count,
            "items": [it.dict() for it in results],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run autolearn: {str(e)}")

@router.get("/azl/autolearn_stream")
async def azl_autolearn_stream(
    topic: str,
    count: int = 10,
    pass_threshold: Optional[float] = None,
    max_attempts: Optional[int] = None,
    judge_model: Optional[str] = None,
    ollama_client: OllamaClient = Depends(get_ollama_client),
    teaching_service: TeachingService = Depends(get_teaching_service),
):
    """Server-Sent Events stream that emits AutoLearn progress live."""
    cfg = get_azl_config()
    pth = pass_threshold if pass_threshold is not None else cfg.pass_threshold
    mat = max_attempts if max_attempts is not None else cfg.max_attempts

    async def _gen():
        import json as _json
        import time as _time
        try:
            # Verify judge model availability up front
            from api.deps import get_judge_ollama_client as _gj
            _judge = _gj()
            # If a judge_model override is provided, rebuild judge client using same host
            if judge_model:
                from agent.llm.ollama_client import OllamaClient as _OC
                from agent.config import get_settings as _gs
                _s = _gs()
                _judge = _OC(host=_s.ollama_host, model=judge_model, timeout_seconds=(_s.ollama_timeout_seconds or 60.0))
            try:
                health = await _judge.health_check()
                if not health.get("ready"):
                    raise RuntimeError(f"Judge model not ready: {health}")
            except Exception as _hc_exc:
                yield _sse_format("backend_error", _json.dumps({"message": f"Judge health check failed: {_hc_exc}"}))
                return
            # Verify main LLM availability up front
            try:
                main_health = await ollama_client.health_check()
                if not main_health.get("ready"):
                    raise RuntimeError(f"Main LLM not ready: {main_health}")
            except Exception as _mh_exc:
                yield _sse_format("backend_error", _json.dumps({"message": f"LLM health check failed: {_mh_exc}"}))
                return
            # include runtime config for transparency
            yield _sse_format("start", _json.dumps({
                "topic": topic,
                "count": count,
                "pass_threshold": pth,
                "max_attempts": mat,
                "weights": cfg.score_weights,
                "judge_model": judge_model or (get_settings().judge_model or get_settings().ollama_model)
            }))
            generator = AZLGenerator(ollama_client)
            examples, proposal_id = await generator.propose_examples(topic, count)
            yield _sse_format("proposed", _json.dumps({"proposal_id": proposal_id, "count": len(examples)}))
            judge_client = _judge
            # Prepare judge fallback candidates and simple circuit breaker
            from agent.config import get_settings as _gs2
            _s2 = _gs2()
            candidate_models: List[str] = []
            if judge_model:
                candidate_models.append(judge_model)
            if _s2.judge_model:
                candidate_models.append(_s2.judge_model)
            if _s2.ollama_model:
                candidate_models.append(_s2.ollama_model)
            # de-duplicate while preserving order
            seen_models = set()
            models_to_try: List[str] = []
            for _m in candidate_models:
                if _m and _m not in seen_models:
                    seen_models.add(_m)
                    models_to_try.append(_m)
            circuit_failures: Dict[str, int] = {}

            accepted = 0
            for idx, ex in enumerate(examples):
                attempts = 0
                best_score = 0.0
                accepted_flag = False
                yield _sse_format("item", _json.dumps({"index": idx, "question": ex.get("question","")}))
                while attempts <= mat:
                    attempts += 1
                    # Announce attempt start so UI shows activity while LLM runs
                    yield _sse_format("attempt", _json.dumps({"index": idx, "attempt": attempts, "max_attempts": mat}))
                    _t0 = _time.perf_counter()
                    # Cascade judge: fast first, then heavy only if near threshold
                    r = None
                    used_model = None
                    last_err: Optional[Exception] = None
                    _qa = f"{ex.get('question','').strip()}||{ex.get('answer','').strip()}"
                    if not hasattr(azl_autolearn_stream, "_judge_cache2"):
                        setattr(azl_autolearn_stream, "_judge_cache2", {})
                    _cache2 = getattr(azl_autolearn_stream, "_judge_cache2")
                    def _cache_key(model: str) -> str:
                        return f"{model}||{_qa}"
                    fast_model = judge_model or (_s2.judge_model or _s2.ollama_model)
                    heavy_model = None  # disabled per current policy
                    margin = float(getattr(_s2, 'judge_margin', 0.1) or 0.1)
                    # --- fast judge
                    try:
                        if _cache2.get(_cache_key(fast_model)):
                            r_fast = _cache2[_cache_key(fast_model)]["result"]
                        else:
                            _sc_fast = AZLScorer(ollama_client, cfg.score_weights, judge_client)
                            r_fast = await _sc_fast.score(ex)
                            _cache2[_cache_key(fast_model)] = {"result": r_fast}
                        used_model = fast_model
                        _ms = int((_time.perf_counter() - _t0) * 1000)
                        yield _sse_format("score", _json.dumps({
                            "index": idx,
                            "attempt": attempts,
                            "score": r_fast.get("score", 0),
                            "checks": r_fast.get("checks", {}),
                            "judge": r_fast.get("judge", {}),
                            "ms": _ms,
                            "judge_model": used_model
                        }))
                    except Exception as _score_exc:
                        last_err = _score_exc
                        r_fast = None
                    # Decide escalation
                    escalate = False
                    if r_fast is None:
                        escalate = bool(heavy_model)
                    else:
                        try:
                            scf = float(r_fast.get("score", 0))
                            escalate = bool(heavy_model) and abs(scf - pth) < margin
                        except Exception:
                            escalate = bool(heavy_model)
                    # --- heavy judge if needed
                    if escalate and heavy_model:
                        try:
                            _jc_heavy = OllamaClient(host=_s2.ollama_host, model=heavy_model, timeout_seconds=(getattr(_s2,'judge_timeout_heavy', 30.0) or 30.0))
                            _sc_heavy = AZLScorer(ollama_client, cfg.score_weights, _jc_heavy)
                            r_heavy = await _sc_heavy.score(ex)
                            _cache2[_cache_key(heavy_model)] = {"result": r_heavy}
                            used_model = heavy_model
                            _ms = int((_time.perf_counter() - _t0) * 1000)
                            yield _sse_format("score", _json.dumps({
                                "index": idx,
                                "attempt": attempts,
                                "score": r_heavy.get("score", 0),
                                "checks": r_heavy.get("checks", {}),
                                "judge": r_heavy.get("judge", {}),
                                "ms": _ms,
                                "judge_model": used_model
                            }))
                            r = r_heavy
                        except Exception as _score_exc2:
                            last_err = _score_exc2
                            r = r_fast
                    else:
                        r = r_fast
                    if r is None:
                        # If even fallback failed, emit error rationale and move on
                        _ms = int((_time.perf_counter() - _t0) * 1000)
                        yield _sse_format("score", _json.dumps({
                            "index": idx,
                            "attempt": attempts,
                            "score": 0,
                            "checks": {},
                            "judge": {"rationale": f"judge_error: {str(last_err) if last_err else 'unknown'}"},
                            "ms": _ms,
                            "judge_model": used_model or fast_model
                        }))
                        # proceed to regeneration
                        best_score = max(best_score, 0.0)
                        # fall through to regen logic below
                    # (Note: emitted score above per stage)
                    try:
                        sc = float(r.get("score", 0))
                        if sc > best_score:
                            best_score = sc
                    except Exception:
                        pass
                    if r and r.get("score", 0) >= pth:
                        accepted += 1
                        teaching_service.store_synthetic_example(ex, f"auto-accepted score={r.get('score',0):.2f}", topic=topic)
                        yield _sse_format("accepted", _json.dumps({"index": idx, "attempts": attempts, "score": r.get("score",0)}))
                        accepted_flag = True
                        break
                    # regenerate
                    current = AZLGenerator.get_example(proposal_id, idx)
                    guidance = "Generate 1 improved question-answer pair for the topic. Keep it precise and factual. Return JSON {\"question\":..., \"answer\":...}."
                    prompt = (
                        f"Topic: {topic or 'general'}\n"
                        f"Current QA to improve:\nQ: {current.get('question','')}\nA: {current.get('answer','')}\n\n{guidance}"
                    )
                    yield _sse_format("regen_attempt", _json.dumps({"index": idx, "attempt": attempts}))
                    # Stronger regeneration instruction for strict JSON
                    sys = (
                        "You are a strict JSON generator for tutoring Q/A. Return ONLY one JSON object with exactly two keys: "
                        "question (string) and answer (string). No prose, no code fences, no extra keys. Example: "
                        "{\"question\": \"What is backpropagation?\", \"answer\": \"Backpropagation is gradient-based weight update...\"}"
                    )
                    _rt0 = _time.perf_counter()
                    _reason = None
                    _snippet = None
                    _regen_ms = None
                    try:
                        regen_resp = await ollama_client.send(prompt=prompt, system_prompt=sys, temperature=0.0)
                        text_out = regen_resp.get("text", "")
                        _regen_ms = int((_time.perf_counter() - _rt0) * 1000)
                        new_ex = AZLGenerator.parse_single_example(text_out)
                        if not new_ex:
                            # last-chance regex JSON extract
                            import re as _re, json as _json2
                            m = _re.search(r"\{[\s\S]*?\}", text_out)
                            if m:
                                try:
                                    cand = _json2.loads(m.group(0))
                                    if isinstance(cand, dict) and "question" in cand and "answer" in cand:
                                        new_ex = {"question": str(cand["question"]).strip(), "answer": str(cand["answer"]).strip()}
                                    else:
                                        _reason = "json_missing_keys"
                                except Exception as _je:
                                    _reason = f"json_error: {_je}"
                            else:
                                _reason = "parse_failed"
                            # capture snippet for debugging
                            try:
                                _snippet = (text_out or "").strip().replace("\n", " ")[:200]
                            except Exception:
                                _snippet = None
                    except Exception as _rexc:
                        _regen_ms = int((_time.perf_counter() - _rt0) * 1000)
                        _reason = f"request_error: {_rexc}"
                        new_ex = None
                    if not new_ex:
                        # continue to next attempt rather than breaking the item loop immediately
                        yield _sse_format("regen_fail", _json.dumps({"index": idx, "attempt": attempts, "reason": _reason, "ms": _regen_ms, "snippet": _snippet}))
                        continue
                    AZLGenerator.set_example(proposal_id, idx, new_ex)
                    ex = new_ex
                    yield _sse_format("regenerated", _json.dumps({"index": idx, "ms": _regen_ms}))
                if not accepted_flag:
                    # Emit explicit failure event so UI can update progress accurately
                    # Include last judge rationale and failing checks if available
                    try:
                        last_checks = r.get("checks", {})
                        last_judge = r.get("judge", {})
                    except Exception:
                        last_checks = {}
                        last_judge = {}
                    yield _sse_format("failed", _json.dumps({
                        "index": idx,
                        "attempts": attempts,
                        "best_score": best_score,
                        "checks": last_checks,
                        "judge": last_judge
                    }))
            yield _sse_format("done", _json.dumps({"accepted": accepted, "failed": max(0, len(examples)-accepted)}))
        except Exception as e:
            import json as _json
            # emit a distinct backend_error event so the frontend can show the reason
            yield _sse_format("backend_error", _json.dumps({"message": str(e)}))

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })

class DashboardDataRequest(BaseModel):
    topic: str = ""

class DashboardDataResponse(BaseModel):
    popular_topics: List[str]
    recent_activities: List[Dict[str, Any]]

@router.post("/dashboard_data", response_model=DashboardDataResponse)
async def get_dashboard_data(
    request: DashboardDataRequest,
    ollama_client: OllamaClient = Depends(get_ollama_client),
):
    """Get dashboard data based on the topic."""
    try:
        # Generate related topics based on the current topic
        prompt = f"""
        Based on the topic "{request.topic or 'general learning'}", generate:
        
        1. A list of 8 related topics that would be popular in a learning platform
        2. A list of 3 recent learning activities with:
           - Activity type (Completed, Mastered, or Started)
           - Topic name
           - Time frame (e.g., "2 days ago", "1 week ago")
        
        Format your response as JSON with the following structure:
        {{
            "popular_topics": ["Topic1", "Topic2", ...],
            "recent_activities": [
                {{"type": "Completed", "topic": "Topic Name", "time": "2 days ago"}},
                ...
            ]
        }}
        """
        
        llm_res = await ollama_client.send(prompt=prompt)
        
        # Extract JSON from response text
        import json as _json
        import re as _re
        raw_text = llm_res.get("text", "")
        match = _re.search(r"```json\s*([\s\S]*?)\s*```", raw_text)
        json_text = match.group(1) if match else raw_text
        # Coerce to valid JSON if needed
        def _coerce(text: str) -> dict:
            try:
                return _json.loads(text)
            except Exception:
                obj = _re.search(r"\{[\s\S]*\}", text)
                if not obj:
                    return {"popular_topics": [], "recent_activities": []}
                candidate = obj.group(0)
                # Remove trailing commas before closing braces/brackets
                candidate = _re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return _json.loads(candidate)
                except Exception:
                    return {"popular_topics": [], "recent_activities": []}
        data = _coerce(json_text)
        
        from fastapi import Response
        return Response(
            content=_json.dumps({
                "popular_topics": data.get("popular_topics", []),
                "recent_activities": data.get("recent_activities", [])
            }),
            media_type="application/json",
            headers={"Cache-Control": "no-store"}
        )
    except Exception as e:
        logger.error(f"dashboard_data failed: {e}")
        from fastapi import Response
        import json as _json
        return Response(
            content=_json.dumps({"popular_topics": [], "recent_activities": []}),
            media_type="application/json",
            headers={"Cache-Control": "no-store"}
        )