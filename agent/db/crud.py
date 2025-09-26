from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy import Select, select, func, desc
from sqlalchemy.orm import Session

from agent.db.models import Interaction, MemoryRule, Session as DbSession, SyntheticExample, User


# Users

def create_user(db: Session, user_id: str, name: Optional[str] = None, preferences: Optional[dict] = None) -> User:
	user = User(user_id=user_id, name=name, preferences=preferences or {})
	db.add(user)
	db.flush()
	return user


def get_user(db: Session, user_id: str) -> Optional[User]:
	stmt: Select[tuple[User]] = select(User).where(User.user_id == user_id)
	return db.execute(stmt).scalars().first()


# Sessions

def create_session(db: Session, session_id: str, user_id: str, topic: str) -> DbSession:
	sess = DbSession(session_id=session_id, user_id=user_id, topic=topic, start_ts=datetime.now(timezone.utc))
	db.add(sess)
	db.flush()
	return sess


def end_session(db: Session, session_id: str, summary: Optional[str] = None) -> Optional[DbSession]:
	sess = db.get(DbSession, session_id)
	if not sess:
		return None
	sess.end_ts = datetime.now(timezone.utc)
	sess.summary = summary
	db.add(sess)
	return sess


def get_session(db: Session, session_id: str) -> Optional[DbSession]:
	return db.get(DbSession, session_id)


# Interactions

def add_interaction(
	db: Session,
	interaction_id: str,
	session_id: str,
	turn_index: int,
	input_text: str,
	agent_response: str,
	agent_strategy_tag: Optional[str],
	outcome: Optional[str],
	confidence_score: Optional[float],
) -> Interaction:
	it = Interaction(
		interaction_id=interaction_id,
		session_id=session_id,
		turn_index=turn_index,
		input_text=input_text,
		agent_response=agent_response,
		agent_strategy_tag=agent_strategy_tag,
		outcome=outcome,
		confidence_score=confidence_score,
	)
	db.add(it)
	db.flush()
	return it


# Memory rules

def upsert_memory_rule(
	db: Session,
	rule_id: str,
	strategy: str,
	user_id: Optional[str] = None,
	success_rate: float = 0.0,
	last_used_ts: Optional[datetime] = None,
) -> MemoryRule:
	rule = db.get(MemoryRule, rule_id)
	if rule is None:
		rule = MemoryRule(
			rule_id=rule_id,
			user_id=user_id,
			strategy=strategy,
			success_rate=success_rate,
			last_used_ts=last_used_ts,
		)
		db.add(rule)
	else:
		rule.strategy = strategy
		rule.user_id = user_id
		rule.success_rate = success_rate
		rule.last_used_ts = last_used_ts
	return rule


def get_memory_rule(db: Session, rule_id: str) -> Optional[MemoryRule]:
	return db.get(MemoryRule, rule_id)


def get_user_memory_rules(db: Session, user_id: str) -> List[MemoryRule]:
	"""Get all memory rules for a user"""
	stmt: Select[tuple[MemoryRule]] = select(MemoryRule).where(MemoryRule.user_id == user_id)
	return list(db.execute(stmt).scalars().all())


# Progress metrics

def get_user_progress_metrics(db: Session, user_id: str) -> Dict[str, Any]:
	"""Get comprehensive progress metrics for a user"""
	# Get total sessions
	sessions_stmt = select(func.count()).select_from(DbSession).where(DbSession.user_id == user_id)
	total_sessions = db.execute(sessions_stmt).scalar() or 0
	
	# Get completed topics (sessions with end_ts)
	completed_stmt = select(func.count()).select_from(DbSession).where(
		DbSession.user_id == user_id,
		DbSession.end_ts.is_not(None)
	)
	topics_completed = db.execute(completed_stmt).scalar() or 0
	
	# Get in-progress topics
	in_progress_stmt = select(DbSession).where(
		DbSession.user_id == user_id,
		DbSession.end_ts.is_(None)
	).order_by(desc(DbSession.start_ts))
	
	in_progress_sessions = list(db.execute(in_progress_stmt).scalars().all())
	topics_in_progress = [
		{
			"topic": session.topic,
			"session_id": session.session_id,
			"start_ts": session.start_ts.isoformat() if session.start_ts else None
		}
		for session in in_progress_sessions
	]
	
	# Calculate accuracy from interactions
	interactions_stmt = select(Interaction).join(DbSession).where(DbSession.user_id == user_id)
	interactions = list(db.execute(interactions_stmt).scalars().all())
	
	total_interactions = len(interactions)
	positive_outcomes = sum(1 for i in interactions if i.outcome == "correct")
	accuracy = (positive_outcomes / total_interactions) * 100 if total_interactions > 0 else 0
	
	# Calculate streak
	streak = 0
	max_streak = 0
	for i in sorted(interactions, key=lambda x: x.created_at):
		if i.outcome == "correct":
			streak += 1
		else:
			streak = 0
		max_streak = max(max_streak, streak)
	
	# Get strategy performance
	strategy_performance = {}
	for i in interactions:
		if not i.agent_strategy_tag:
			continue
		
		strategy = i.agent_strategy_tag
		if strategy not in strategy_performance:
			strategy_performance[strategy] = {"attempts": 0, "correct": 0}
		
		strategy_performance[strategy]["attempts"] += 1
		if i.outcome == "correct":
			strategy_performance[strategy]["correct"] += 1
	
	# Calculate success rates
	for strategy, stats in strategy_performance.items():
		attempts = stats["attempts"]
		correct = stats["correct"]
		stats["success_rate"] = (correct / attempts) * 100 if attempts > 0 else 0
	
	return {
		"total_sessions": total_sessions,
		"topics_completed": topics_completed,
		"topics_in_progress": topics_in_progress,
		"total_interactions": total_interactions,
		"accuracy": round(accuracy, 1),
		"streak": max_streak,
		"strategy_performance": strategy_performance
	}


def get_session_progress_metrics(db: Session, session_id: str) -> Dict[str, Any]:
	"""Get progress metrics for a specific session"""
	# Get session info
	session = db.get(DbSession, session_id)
	if not session:
		return {
			"accuracy": 0.0,
			"streak": 0,
			"topics_completed": 0,
			"topics_in_progress": [],
			"error": "Session not found"
		}
	
	# Get interactions for this session
	interactions_stmt = select(Interaction).where(Interaction.session_id == session_id).order_by(Interaction.turn_index)
	interactions = list(db.execute(interactions_stmt).scalars().all())
	
	total_interactions = len(interactions)
	positive_outcomes = sum(1 for i in interactions if i.outcome == "correct")
	accuracy = (positive_outcomes / total_interactions) * 100 if total_interactions > 0 else 0
	
	# Calculate current streak
	streak = 0
	for i in reversed(interactions):  # Start from most recent
		if i.outcome == "correct":
			streak += 1
		else:
			break  # Stop counting at first incorrect
	
	# Get user's completed topics
	completed_stmt = select(func.count()).select_from(DbSession).where(
		DbSession.user_id == session.user_id,
		DbSession.end_ts.is_not(None)
	)
	topics_completed = db.execute(completed_stmt).scalar() or 0
	
	# Get current topic
	topic_in_progress = [{
		"topic": session.topic,
		"session_id": session_id,
		"start_ts": session.start_ts.isoformat() if session.start_ts else None,
		"accuracy": accuracy
	}]
	
	# Get current strategy
	current_strategy = "neural_compression"  # Default
	if interactions:
		latest = interactions[-1]
		if latest.agent_strategy_tag:
			current_strategy = latest.agent_strategy_tag
	
	# Determine difficulty level based on accuracy
	if accuracy >= 80:
		current_difficulty = "advanced"
	elif accuracy >= 50:
		current_difficulty = "intermediate"
	else:
		current_difficulty = "beginner"
	
	# Calculate mastery level (0-10)
	mastery_level = min(10, max(0, int(accuracy / 10)))
	
	return {
		"accuracy": round(accuracy, 1),
		"streak": streak,
		"topics_completed": topics_completed,
		"topics_in_progress": topic_in_progress,
		"total_interactions": total_interactions,
		"positive_outcomes": positive_outcomes,
		"current_strategy": current_strategy,
		"current_difficulty": current_difficulty,
		"mastery_level": mastery_level
	}


# Synthetic examples

def create_synthetic_example(
	db: Session,
	example_id: str,
	source: str,
	payload: dict,
	validator_passed: bool,
	verified_by: Optional[str] = None,
) -> SyntheticExample:
	ex = SyntheticExample(
		example_id=example_id,
		source=source,
		payload=payload,
		validator_passed=validator_passed,
		verified_by=verified_by,
	)
	db.add(ex)
	db.flush()
	return ex

