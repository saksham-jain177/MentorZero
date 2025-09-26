from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
	Boolean,
	DateTime,
	Float,
	ForeignKey,
	Integer,
	JSON,
	String,
	Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
	pass


class User(Base):
	"""User profile and preferences."""

	__tablename__ = "users"

	user_id: Mapped[str] = mapped_column(String, primary_key=True)
	name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
	preferences: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
	)

	sessions: Mapped[list[Session]] = relationship(back_populates="user", cascade="all, delete-orphan")  # type: ignore[name-defined]
	memory_rules: Mapped[list[MemoryRule]] = relationship(back_populates="user", cascade="all, delete-orphan")  # type: ignore[name-defined]


class Session(Base):
	"""Teaching session for a user on a topic."""

	__tablename__ = "sessions"

	session_id: Mapped[str] = mapped_column(String, primary_key=True)
	user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False)
	topic: Mapped[str] = mapped_column(String, nullable=False)
	start_ts: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
	)
	end_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
	summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

	user: Mapped[User] = relationship(back_populates="sessions")
	interactions: Mapped[list[Interaction]] = relationship(
		back_populates="session", cascade="all, delete-orphan"
	)  # type: ignore[name-defined]


class Interaction(Base):
	"""One turn of interaction during a session."""

	__tablename__ = "interactions"

	interaction_id: Mapped[str] = mapped_column(String, primary_key=True)
	session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), nullable=False)
	turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
	input_text: Mapped[str] = mapped_column(Text, nullable=False)
	agent_response: Mapped[str] = mapped_column(Text, nullable=False)
	agent_strategy_tag: Mapped[Optional[str]] = mapped_column(String, nullable=True)
	outcome: Mapped[Optional[str]] = mapped_column(String, nullable=True)
	confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
	)

	session: Mapped[Session] = relationship(back_populates="interactions")


class MemoryRule(Base):
	"""Per-user strategy rule and performance."""

	__tablename__ = "memory_rules"

	rule_id: Mapped[str] = mapped_column(String, primary_key=True)
	user_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.user_id"), nullable=True)
	strategy: Mapped[str] = mapped_column(Text, nullable=False)
	success_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
	last_used_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

	user: Mapped[Optional[User]] = relationship(back_populates="memory_rules")


class SyntheticExample(Base):
	"""Synthetic or human examples proposed/validated for training data."""

	__tablename__ = "synthetic_examples"

	example_id: Mapped[str] = mapped_column(String, primary_key=True)
	source: Mapped[str] = mapped_column(String, nullable=False)
	payload: Mapped[dict] = mapped_column(JSON, nullable=False)
	validator_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
	verified_by: Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.user_id"), nullable=True)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
	)


class EvaluationCache(Base):
	"""Cache for AZL evaluation results to avoid redundant LLM calls."""

	__tablename__ = "evaluation_cache"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	example_hash: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
	topic: Mapped[str] = mapped_column(String, index=True, nullable=False)
	fast_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
	reasoning_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
	confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
	method: Mapped[str] = mapped_column(String, nullable=False)  # 'fast' or 'reasoning'
	checks: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
	judge_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
	)
	expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PerformanceMetrics(Base):
	"""Performance metrics for monitoring system health and optimization."""

	__tablename__ = "performance_metrics"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
	metric_type: Mapped[str] = mapped_column(String, index=True, nullable=False)  # 'azl_generation', 'azl_evaluation', 'rag_retrieval', etc.
	operation: Mapped[str] = mapped_column(String, nullable=False)  # specific operation name
	duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
	success: Mapped[bool] = mapped_column(Boolean, nullable=False)
	model_used: Mapped[Optional[str]] = mapped_column(String, nullable=True)
	extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # renamed from metadata to avoid conflict
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
	)

