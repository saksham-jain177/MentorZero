import sqlite3
import json
import uuid
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GraphStore:
    """
    SQLite-backed store for knowledge graph persistence.
    Manages entities (nodes) and their relationships (edges).
    """
    
    def __init__(self, db_path: str = "./data/mentorzero.db"):
        self.db_path = db_path
        self._ensure_db_exists()
        self._init_db()
        
    def _ensure_db_exists(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _init_db(self):
        """Initialize database with schema if not already present"""
        try:
            # Read schema file
            schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
            with open(schema_path, "r") as f:
                schema_script = f.read()
                
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_script)
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            
    def add_node(self, name: str, node_type: str, metadata: Optional[Dict] = None) -> str:
        """Add or update a node in the graph"""
        node_id = hashlib_id(name.lower())
        metadata_json = json.dumps(metadata or {})
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO nodes (id, name, type, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        type = excluded.type,
                        metadata = excluded.metadata,
                        updated_at = excluded.updated_at
                """, (node_id, name, node_type, metadata_json, datetime.now().isoformat()))
            return node_id
        except Exception as e:
            logger.error(f"Error adding node {name}: {e}")
            return node_id

    def add_edge(self, source_id: str, target_id: str, relation: str, metadata: Optional[Dict] = None):
        """Add a relationship between nodes"""
        edge_id = hashlib_id(f"{source_id}:{target_id}:{relation}")
        metadata_json = json.dumps(metadata or {})
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO edges (id, source_id, target_id, relation, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO NOTHING
                """, (edge_id, source_id, target_id, relation, metadata_json))
        except Exception as e:
            logger.error(f"Error adding edge {source_id}->{target_id}: {e}")

    def get_full_graph(self) -> Dict[str, List]:
        """Retrieve all nodes and edges for visualization"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Fetch nodes
                nodes_rows = conn.execute("SELECT * FROM nodes").fetchall()
                nodes = []
                for row in nodes_rows:
                    nodes.append({
                        "id": row["id"],
                        "name": row["name"],
                        "type": row["type"],
                        "metadata": json.loads(row["metadata"] or "{}")
                    })
                    
                # Fetch edges
                edges_rows = conn.execute("SELECT * FROM edges").fetchall()
                edges = []
                for row in edges_rows:
                    edges.append({
                        "id": row["id"],
                        "source": row["source_id"],
                        "target": row["target_id"],
                        "relation": row["relation"],
                        "metadata": json.loads(row["metadata"] or "{}")
                    })
                    
                return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Error retrieving full graph: {e}")
            return {"nodes": [], "edges": []}

    def search_subgraph(self, query: str) -> Dict[str, List]:
        """Search for a specific entity and its neighbors"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Find matching nodes
                search_query = f"%{query}%"
                match_rows = conn.execute(
                    "SELECT id FROM nodes WHERE name LIKE ? OR type LIKE ?", 
                    (search_query, search_query)
                ).fetchall()
                
                match_ids = [row["id"] for row in match_rows]
                if not match_ids:
                    return {"nodes": [], "edges": []}
                
                # Get those nodes and their first-degree edges
                placeholders = ",".join(["?"] * len(match_ids))
                
                # Get nodes
                nodes_rows = conn.execute(
                    f"SELECT * FROM nodes WHERE id IN ({placeholders})", 
                    match_ids
                ).fetchall()
                
                # Get edges (connected to these nodes)
                edges_rows = conn.execute(f"""
                    SELECT * FROM edges 
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                """, match_ids + match_ids).fetchall()
                
                # Also collect neighbor node IDs that were not in the initial match
                neighbor_ids = set()
                for row in edges_rows:
                    neighbor_ids.add(row["source_id"])
                    neighbor_ids.add(row["target_id"])
                
                all_node_ids = list(set(match_ids) | neighbor_ids)
                all_placeholders = ",".join(["?"] * len(all_node_ids))
                
                # Re-fetch all relevant nodes
                final_nodes_rows = conn.execute(
                    f"SELECT * FROM nodes WHERE id IN ({all_placeholders})", 
                    all_node_ids
                ).fetchall()
                
                nodes = [dict(row) for row in final_nodes_rows]
                for n in nodes:
                    n["metadata"] = json.loads(n["metadata"] or "{}")
                    
                edges = [dict(row) for row in edges_rows]
                for e in edges:
                    e["metadata"] = json.loads(e["metadata"] or "{}")
                    # Rename for Cytoscape compatibility
                    e["source"] = e.pop("source_id")
                    e["target"] = e.pop("target_id")
                    
                return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Error searching subgraph for {query}: {e}")
            return {"nodes": [], "edges": []}

def hashlib_id(text: str) -> str:
    """Generate a consistent hash ID for a string"""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()
