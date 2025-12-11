# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HIPAA RAG is a Retrieval-Augmented Generation solution for querying HIPAA documentation (Parts 160, 162, 164). It parses a PDF, indexes content with vector embeddings, and provides a chat interface for compliance questions with citations.

## Commands

### Run the Full Stack
```bash
docker compose up --build
```

### Run with Public Tunnel (Cloudflare)
```bash
docker compose --profile tunnel up --build
```

### Run Backend Locally (Development)
```bash
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### Run Frontend Locally
```bash
cd frontend
python app.py
```

## Architecture

### Container Stack
- **db**: PostgreSQL 16 with pgvector extension (port 5432)
- **backend**: FastAPI async service (port 8000)
- **frontend**: Gradio chat UI (port 7860)
- **nginx**: Reverse proxy on port 80, routes `/` to frontend, `/api/` to backend, `/health` to backend
- **tunnel**: Optional Cloudflare tunnel for public access (profile: `tunnel`)

### Backend Structure (`backend/app/`)
- `main.py` - FastAPI app with CORS, lifespan management, health/root endpoints
- `config.py` - Pydantic settings from environment variables
- `database.py` - SQLAlchemy async setup with pgvector `Document` and `ChatHistory` models
- `models.py` - Pydantic request/response schemas (`ChatRequest`, `ChatResponse`, `IngestRequest`, `IngestResponse`, `HealthResponse`, `SourceReference`)
- `routers/chat.py` - `/api/chat` endpoint, `/api/chat/sections/{section_ref}` for section lookup
- `routers/ingest.py` - `/api/ingest` for PDF parsing, `/api/ingest/status` for index status
- `services/pdf_parser.py` - `HIPAAParser` extracts sections using regex for parts, subparts, sections (§)
- `services/embeddings.py` - Singleton `EmbeddingService` using sentence-transformers
- `services/retrieval.py` - `RetrievalService` for vector similarity search via pgvector cosine distance
- `services/llm.py` - `LLMService` with Anthropic Claude client and system prompt enforcing citations

### Frontend Structure (`frontend/`)
- `app.py` - Gradio Blocks app with chat interface, ingestion controls, status display

### Data Flow
1. **Ingest**: PDF -> `HIPAAParser` chunks with metadata -> `EmbeddingService` generates vectors -> PostgreSQL stores chunks with embeddings
2. **Query**: User message -> embed query -> pgvector similarity search (top-k) -> format context -> Claude generates answer with citations

## Configuration

### Required Environment Variables
| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |

### Optional Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://hipaa:hipaa_secure_pass@localhost:5432/hipaa_rag` | PostgreSQL connection string |
| `POSTGRES_USER` | `hipaa` | PostgreSQL username (docker-compose) |
| `POSTGRES_PASSWORD` | `hipaa_secure_pass` | PostgreSQL password (docker-compose) |
| `POSTGRES_DB` | `hipaa_rag` | PostgreSQL database name (docker-compose) |
| `CLOUDFLARE_TUNNEL_TOKEN` | - | Cloudflare tunnel token for public access |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL for frontend container |

### Application Settings (via `config.py`)
| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `embedding_dimension` | `384` | Vector embedding size |
| `chunk_size` | `800` | Max characters per chunk |
| `chunk_overlap` | `100` | Overlap between chunks |
| `retrieval_top_k` | `8` | Number of chunks to retrieve |
| `llm_model` | `claude-sonnet-4-20250514` | Anthropic model for responses |
| `llm_max_tokens` | `2048` | Max tokens in LLM response |
| `pdf_path` | `/app/data/HIPAA_questions.pdf` | Default PDF path |

## Database Schema

### `documents` table
| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL PRIMARY KEY` | Auto-increment ID |
| `content` | `TEXT NOT NULL` | Chunk text with context header |
| `section_reference` | `VARCHAR(255)` | Formatted ref: "Part \| Subpart \| Section" |
| `part_number` | `VARCHAR(50)` | HIPAA part (e.g., "164") |
| `section_number` | `VARCHAR(50)` | Section number (e.g., "164.308") |
| `paragraph_reference` | `VARCHAR(100)` | Paragraph ref (not currently populated) |
| `page_number` | `INTEGER` | Primary page number |
| `parent_context` | `TEXT` | Section title for context |
| `embedding` | `vector(384)` | pgvector embedding |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | Creation timestamp |

**Indexes**: `ivfflat` on embedding (cosine), btree on `section_reference`, `part_number`

### `chat_history` table
| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL PRIMARY KEY` | Auto-increment ID |
| `session_id` | `VARCHAR(255) NOT NULL` | Session identifier |
| `role` | `VARCHAR(50) NOT NULL` | Message role (user/assistant) |
| `content` | `TEXT NOT NULL` | Message content |
| `created_at` | `TIMESTAMP WITH TIME ZONE` | Creation timestamp |

**Note**: `chat_history` is defined but not currently used in the chat flow.

## API Endpoints

### Chat
- `POST /api/chat` - Send message, get answer with sources
  - Request: `{ "message": string, "session_id"?: string }`
  - Response: `{ "answer": string, "sources": SourceReference[], "session_id": string }`
- `GET /api/chat/sections/{section_ref}` - Retrieve chunks matching section reference

### Ingestion
- `POST /api/ingest` - Parse and index the HIPAA PDF (clears existing data)
  - Request: `{ "pdf_path"?: string }` (optional, uses default if omitted)
  - Response: `{ "status": string, "chunks_processed": int, "message": string }`
- `GET /api/ingest/status` - Check indexing status
  - Response: `{ "indexed_chunks": int, "status": "ready"|"not_indexed", "pdf_path": string, "pdf_exists": bool }`

### System
- `GET /health` - Health check with database and model info
- `GET /` - Service info with links to docs

## Project Files

```
hipaa-rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py
│   │   │   └── ingest.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── embeddings.py
│   │       ├── llm.py
│   │       ├── pdf_parser.py
│   │       └── retrieval.py
│   ├── data/           # Mount for PDF files
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   └── Dockerfile
├── nginx/
│   └── nginx.conf
├── scripts/
│   └── init_db.sql
├── docker-compose.yml
├── .env.example
└── CLAUDE.md
```
