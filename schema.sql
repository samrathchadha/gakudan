-- Sessions table
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  client_id TEXT NOT NULL,
  prompt TEXT NOT NULL,
  status TEXT NOT NULL,
  contract_status TEXT NOT NULL DEFAULT 'pending',
  created_at DOUBLE PRECISION NOT NULL,
  last_updated DOUBLE PRECISION NOT NULL,
  error TEXT,
  contract_error TEXT
);

-- Graph files table
CREATE TABLE graph_files (
  id SERIAL PRIMARY KEY,
  session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  iteration INTEGER NOT NULL,
  created_at DOUBLE PRECISION NOT NULL,
  data JSONB NOT NULL
);

-- Client tracking table
CREATE TABLE client_tracking (
  client_id TEXT PRIMARY KEY,
  session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
  last_file TEXT,
  last_check DOUBLE PRECISION NOT NULL
);

-- Final response table for storing final outputs
CREATE TABLE final_responses (
  id SERIAL PRIMARY KEY,
  session_id TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
  data JSONB NOT NULL,
  created_at DOUBLE PRECISION NOT NULL
);

-- Create indexes
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_graph_files_session_id ON graph_files(session_id);
CREATE INDEX idx_client_tracking_session_id ON client_tracking(session_id);
CREATE INDEX idx_final_responses_session_id ON final_responses(session_id);
