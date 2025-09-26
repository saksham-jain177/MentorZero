from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent.config import get_settings


_engine = None
_SessionLocal = None


def get_engine():
	global _engine
	if _engine is None:
		s = get_settings()
		db_url = f"sqlite+pysqlite:///{s.db_path}" if s.db_path else "sqlite+pysqlite:///:memory:"
		_engine = create_engine(db_url, connect_args={"check_same_thread": False})
	return _engine


def get_session_factory():
	global _SessionLocal
	if _SessionLocal is None:
		_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
	return _SessionLocal


@contextmanager

def session_scope() -> Iterator:
	"""Provide a transactional scope around a series of operations."""
	SessionLocal = get_session_factory()
	session = SessionLocal()
	try:
		yield session
		session.commit()
	except Exception:  # noqa: BLE001
		session.rollback()
		raise
	finally:
		session.close()

