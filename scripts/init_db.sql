-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for storing parsed PDF chunks
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    section_reference VARCHAR(255),
    part_number VARCHAR(50),
    section_number VARCHAR(50),
    paragraph_reference VARCHAR(100),
    page_number INTEGER,
    parent_context TEXT,
    embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create index for section lookups
CREATE INDEX IF NOT EXISTS documents_section_idx ON documents (section_reference);
CREATE INDEX IF NOT EXISTS documents_part_idx ON documents (part_number);

-- Create chat history table (optional, for session persistence)
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS chat_history_session_idx ON chat_history (session_id);
