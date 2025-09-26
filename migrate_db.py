"""
Database migration script to create new tables for performance optimization.
"""
from sqlalchemy import create_engine
from agent.db.models import Base, EvaluationCache, PerformanceMetrics
from agent.config import get_settings

def migrate():
    """Run database migrations."""
    settings = get_settings()
    
    # Create engine
    engine = create_engine(f"sqlite:///{settings.db_path}")
    
    # Create all tables (will skip existing ones)
    Base.metadata.create_all(bind=engine)
    
    print(f"✅ Database migrated successfully at {settings.db_path}")
    print("✅ New tables created:")
    print("  - evaluation_cache (for caching AZL evaluations)")
    print("  - performance_metrics (for tracking system performance)")

if __name__ == "__main__":
    migrate()