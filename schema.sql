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

-- Nodes table
CREATE TABLE nodes (
  node_id TEXT PRIMARY KEY,
  session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
  prompt TEXT,
  response TEXT,
  depth INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  attributes JSONB -- For any additional node attributes
);

-- Links table
CREATE TABLE links (
  id SERIAL PRIMARY KEY,
  session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
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
  session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
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
