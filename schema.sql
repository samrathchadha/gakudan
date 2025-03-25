-- Drop existing tables in reverse order of dependencies
DROP TABLE IF EXISTS contract_results CASCADE;
DROP TABLE IF EXISTS client_tracking CASCADE;
DROP TABLE IF EXISTS links CASCADE;
DROP TABLE IF EXISTS nodes CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;

-- Sessions table
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  client_id TEXT NOT NULL,
  prompt TEXT NOT NULL,
  status TEXT NOT NULL,
  contract_status TEXT NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL,
  last_updated TIMESTAMPTZ NOT NULL,
  error TEXT,
  contract_error TEXT
);

-- Nodes table with composite primary key
CREATE TABLE nodes (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  node_id TEXT NOT NULL,
  prompt TEXT,
  response TEXT,
  depth INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  attributes JSONB,
  PRIMARY KEY (session_id, node_id)
);

-- Links table
CREATE TABLE links (
  id SERIAL PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  source_id TEXT NOT NULL, 
  target_id TEXT NOT NULL,
  edge_type TEXT, -- 'hierarchy' or 'rag'
  link_type TEXT, -- For compatibility with the existing code
  similarity FLOAT, -- For RAG connections
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  attributes JSONB -- For any additional link attributes
);

-- Client tracking table
CREATE TABLE client_tracking (
  client_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  last_check TIMESTAMPTZ NOT NULL
);

-- Contract results
CREATE TABLE contract_results (
  session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_nodes_session_id ON nodes(session_id);
CREATE INDEX idx_nodes_updated_at ON nodes(updated_at);
CREATE INDEX idx_links_session_id ON links(session_id);
CREATE INDEX idx_links_created_at ON links(created_at);
CREATE INDEX idx_links_source_target ON links(source_id, target_id);
