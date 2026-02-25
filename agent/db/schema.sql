-- Knowledge Graph Schema for MentorZero

-- Nodes table
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    metadata TEXT, -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Edges table (relationships)
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    metadata TEXT, -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES nodes (id),
    FOREIGN KEY (target_id) REFERENCES nodes (id)
);

-- Index for faster source/target lookups
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

-- Index for node name searches
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);

-- Research Sessions table for history and persistence
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    results_json TEXT, -- Full JSON output of the research
    niche_focus TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for session query searches
CREATE INDEX IF NOT EXISTS idx_sessions_query ON research_sessions(query);
