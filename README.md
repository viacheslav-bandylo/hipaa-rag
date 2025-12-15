# HIPAA RAG

A Retrieval-Augmented Generation (RAG) solution for querying HIPAA documentation. This system parses HIPAA regulations (Parts 160, 162, 164), indexes content with vector embeddings, and provides a chat interface for compliance questions with accurate citations.

## Features

- **PDF Parsing**: Automated extraction of HIPAA sections with metadata (part numbers, section references, page numbers)
- **Hybrid Search**: Combines vector similarity search (pgvector) with BM25 keyword matching for improved retrieval
- **AI-Powered Answers**: Uses Claude (Anthropic) to generate accurate responses with citations to specific HIPAA sections
- **Citation Tracking**: Every answer includes references to source sections and page numbers
- **Web Interface**: User-friendly Gradio chat UI for asking compliance questions
- **Docker Deployment**: Full containerized stack with PostgreSQL, FastAPI backend, and Gradio frontend
- **Public Access**: Optional Cloudflare tunnel for secure public deployment

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────▶│  Frontend   │────▶│  Backend    │
│   (Port 80) │     │  (Gradio)   │     │  (FastAPI)  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ PostgreSQL  │
                                        │ + pgvector  │
                                        └─────────────┘
```

### Components

| Service | Description | Port |
|---------|-------------|------|
| **db** | PostgreSQL 16 with pgvector extension | 5432 |
| **backend** | FastAPI async API service | 8000 |
| **frontend** | Gradio chat interface | 7860 |
| **nginx** | Reverse proxy | 80 |
| **tunnel** | Cloudflare tunnel (optional) | - |

## Prerequisites

- Docker and Docker Compose
- [Anthropic API key](https://console.anthropic.com/)
- HIPAA PDF document (place in `backend/data/`)

## Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd hipaa-rag
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

3. **Add HIPAA PDF**

   Place your HIPAA PDF file at `backend/data/HIPAA_questions.pdf`

4. **Start the stack**

   ```bash
   docker compose up --build
   ```

5. **Access the application**

   - Web UI: http://localhost (or http://localhost:7860)
   - API Docs: http://localhost/api/docs
   - Health Check: http://localhost/health

6. **Ingest documents**

   Click "Ingest Documents" in the web UI or call the API:

   ```bash
   curl -X POST http://localhost/api/ingest
   ```

## Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://hipaa:hipaa_secure_pass@localhost:5432/hipaa_rag` | PostgreSQL connection string |
| `POSTGRES_USER` | `hipaa` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `hipaa_secure_pass` | PostgreSQL password |
| `POSTGRES_DB` | `hipaa_rag` | PostgreSQL database name |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL for frontend |
| `CLOUDFLARE_TUNNEL_TOKEN` | - | Cloudflare tunnel token for public access |

### Application Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `embedding_dimension` | `384` | Vector embedding size |
| `chunk_size` | `800` | Max characters per chunk |
| `chunk_overlap` | `100` | Overlap between chunks |
| `retrieval_top_k` | `8` | Number of chunks to retrieve |
| `llm_model` | `claude-sonnet-4-20250514` | Claude model for responses |
| `llm_max_tokens` | `2048` | Max tokens in LLM response |

## API Endpoints

### Chat

- **POST** `/api/chat` - Send a message and get an AI-generated answer with sources

  ```json
  {
    "message": "What is protected health information?",
    "session_id": "optional-session-id"
  }
  ```

- **GET** `/api/chat/sections/{section_ref}` - Retrieve chunks matching a section reference

### Ingestion

- **POST** `/api/ingest` - Parse and index the HIPAA PDF

  ```json
  {
    "pdf_path": "/app/data/HIPAA_questions.pdf"
  }
  ```

- **GET** `/api/ingest/status` - Check indexing status

### System

- **GET** `/health` - Health check with database and model info
- **GET** `/` - Service info with links to documentation

## Development

### Run Backend Locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Run Frontend Locally

```bash
cd frontend
pip install -r requirements.txt
python app.py
```

### Run with Public Tunnel (Cloudflare)

To make your HIPAA RAG application publicly accessible without exposing ports or configuring a static IP, use Cloudflare Tunnels.

#### 1. Create a Cloudflare Tunnel

1. Sign up for [Cloudflare](https://cloudflare.com/) and add a domain (free plan works)
2. Go to [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/)
3. Navigate to **Networks** → **Tunnels**
4. Click **Create a tunnel**
5. Select **Cloudflared** as the connector type
6. Name your tunnel (e.g., `hipaa-rag`)
7. Copy the tunnel token provided

#### 2. Configure the Token

Add the token to your `.env` file:

```bash
CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoiYWJjZGVmLi4uIiwidCI6Ii4uLiIsInMiOiIuLi4ifQ==
```

#### 3. Configure Public Hostname

In the Cloudflare Zero Trust dashboard, configure the tunnel's public hostname:

1. Go to your tunnel's configuration
2. Add a **Public Hostname**:
   - **Subdomain**: your choice (e.g., `hipaa-rag`)
   - **Domain**: select your Cloudflare domain
   - **Service Type**: HTTP
   - **URL**: `nginx:80`

#### 4. Start with Tunnel Profile

```bash
docker compose --profile tunnel up --build
```

Your application will be accessible at `https://hipaa-rag.yourdomain.com` (or whatever subdomain you configured).

#### How It Works

```
Internet → Cloudflare Edge → Tunnel Container → Nginx → Frontend/Backend
```

The tunnel creates an outbound-only connection to Cloudflare, so no inbound ports need to be opened on your firewall.

## Project Structure

```
hipaa-rag/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Settings management
│   │   ├── database.py          # SQLAlchemy models
│   │   ├── models.py            # Pydantic schemas
│   │   ├── routers/
│   │   │   ├── chat.py          # Chat endpoints
│   │   │   └── ingest.py        # Ingestion endpoints
│   │   └── services/
│   │       ├── embeddings.py    # Embedding service
│   │       ├── llm.py           # Claude LLM service
│   │       ├── pdf_parser.py    # HIPAA PDF parser
│   │       ├── retrieval.py     # Vector search service
│   │       └── bm25_service.py  # BM25 keyword search
│   ├── data/                    # PDF files directory
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Gradio application
│   ├── Dockerfile
│   └── requirements.txt
├── nginx/
│   └── nginx.conf               # Reverse proxy config
├── scripts/
│   └── init_db.sql              # Database initialization
├── docker-compose.yml
├── .env.example
├── CLAUDE.md
└── README.md
```

## Data Flow

1. **Ingestion Pipeline**
   - PDF uploaded → `HIPAAParser` extracts sections with metadata
   - Text chunked with overlap → `EmbeddingService` generates vectors
   - Chunks stored in PostgreSQL with pgvector embeddings
   - BM25 index built for keyword search

2. **Query Pipeline**
   - User submits question → Query embedded via sentence-transformers
   - Hybrid retrieval: pgvector similarity + BM25 keyword matching
   - Top-k relevant chunks formatted as context
   - Claude generates answer with citations to specific sections

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy (async), Pydantic
- **Database**: PostgreSQL 16 with pgvector extension
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Search**: pgvector (vector similarity) + rank-bm25 (keyword)
- **LLM**: Anthropic Claude
- **Frontend**: Gradio
- **Infrastructure**: Docker, Nginx, Cloudflare Tunnels
