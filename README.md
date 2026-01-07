# Semantic Image Search

A production-ready pipeline and service for semantic image search using embeddings and vector databases.

## Project Objective

Semantic Image Search provides an end-to-end system to:
- **Ingest** images and generate semantic embeddings from vision-language models
- **Store** vectors efficiently in a vector database (Qdrant)
- **Retrieve** semantically similar images based on text or image queries
- **Evaluate** retrieval quality with recall metrics
- **Serve** results through REST APIs and a web UI

This enables applications to find visually and semantically similar images at scale.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12 |
| **Vector Database** | Qdrant (vector storage and similarity search) |
| **ML/Embeddings** | Vision-language models for image captioning and embeddings |
| **Backend API** | FastAPI (backend service layer) |
| **Frontend/UI** | Streamlit (web interface in `ui/app.py`) |
| **Storage** | Local filesystem + S3-compatible storage (configurable) |
| **Data Processing** | Pandas, NumPy |
| **Testing** | Pytest |
| **Infrastructure** | Docker, Docker Compose, Kubernetes manifests, GitHub Actions CI |
| **Config Management** | YAML-based configs (logging, Qdrant, S3, settings) |
| **Dependency Management** | Poetry (pyproject.toml) + pip (requirements.txt) |

## Directory Structure

```
semantic-image-search/
├── semantic_image_search/          # Main package
│   ├── backend/                    # Core business logic
│   │   ├── main.py                 # Backend service entry point
│   │   ├── ingestion.py            # Image ingestion pipeline
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── retriever.py            # Query-to-result retrieval
│   │   ├── qdrant_client.py        # Qdrant vector DB client
│   │   ├── query_translator.py     # Query processing
│   │   ├── config.py               # Backend configuration
│   │   ├── logger/                 # Logging utilities
│   │   └── exception/              # Custom exceptions
│   ├── src/                        # Higher-level app wiring
│   │   ├── api/                    # API route definitions
│   │   ├── core/                   # Core domain logic
│   │   ├── models/                 # Data models
│   │   ├── services/               # Business services
│   │   ├── storage/                # Storage abstraction
│   │   ├── vectorstore/            # Vector store interface
│   │   ├── monitoring/             # Health checks, metrics
│   │   ├── tasks/                  # Background tasks
│   │   └── utils/                  # Utility functions
│   ├── notebooks/                  # Jupyter notebooks
│   │   ├── evaluate_recall_at_k.ipynb    # Recall evaluation
│   │   ├── experiments.ipynb             # General experiments
│   │   ├── model_comparison.ipynb        # Model benchmarking
│   │   ├── images/                       # Notebook assets
│   │   └── retrieved_results/            # Query results storage
│   ├── tests/                      # Unit and integration tests
│   │   ├── test_api.py
│   │   ├── test_embeddings.py
│   │   ├── test_ingestion.py
│   │   ├── test_vectorstore.py
│   │   └── test_storage.py
│   ├── ui/                         # Web interface
│   │   └── app.py                  # Streamlit app
│   ├── config/                     # Configuration files (YAML)
│   │   ├── logging.yaml
│   │   ├── qdrant.yaml
│   │   ├── s3.yaml
│   │   └── settings.yaml
│   ├── infra/                      # Infrastructure as Code
│   │   ├── Dockerfile              # Container image
│   │   ├── docker-compose.yml      # Local dev stack (Qdrant, app)
│   │   ├── k8s/                    # Kubernetes manifests
│   │   ├── nginx/                  # Reverse proxy config
│   │   └── github-actions/         # CI/CD workflows
│   └── README.md
├── images/                         # Sample image dataset
│   ├── animal/
│   ├── flower/
│   ├── furniture/
│   ├── general/
│   ├── uncategorized/
│   └── weapon/
├── data/
│   └── query_images/               # Query test images
├── env/                            # Virtual environment (if using local venv)
├── main.py                         # Top-level project runner
├── get_lib_versions.py             # Print installed package versions
├── project_structure.py            # Print project tree structure
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Poetry configuration
└── README.md                       # This file
```

## Quick Start

### Prerequisites
- Python 3.12+
- Qdrant instance (local Docker or remote server)

### 1. Setup Environment

```bash
# Create virtual environment (if not already in env/)
python -m venv venv

# Activate it
venv\Scripts\activate              # Windows
# or
source venv/bin/activate           # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant (Local Development)

```bash
# Using Docker Compose
cd semantic_image_search/infra
docker-compose up -d

# Qdrant will be available at http://localhost:6333
```

### 3. Ingest Images

```bash
# Index sample images into Qdrant
python semantic_image_search/backend/ingestion.py \
  --images images/ \
  --config semantic_image_search/config/qdrant.yaml
```

### 4. Run Backend Service

```bash
# Start FastAPI backend (defaults to http://localhost:8000)
uvicorn semantic_image_search.backend.main:app --reload
```

### 5. Run Web UI

```bash
# Launch Streamlit app (defaults to http://localhost:8501)
streamlit run semantic_image_search/ui/app.py
```

## Important Commands

### Development

| Command | Purpose |
|---------|---------|
| `python main.py` | Run top-level project script |
| `python get_lib_versions.py` | Add installed package versions to requirements.txt |
| `python project_structure.py` | Create project directory with files |

### Backend & Ingestion

| Command | Purpose |
|---------|---------|
| `python semantic_image_search/backend/main.py` | Start backend API server |
| `python semantic_image_search/backend/ingestion.py --images <path> --config <yaml>` | Ingest and index images |
| `python semantic_image_search/backend/retriever.py --query <text>` | Run a single query (CLI) |

### Frontend

| Command | Purpose |
|---------|---------|
| `streamlit run semantic_image_search/ui/app.py` | Launch web UI for searching |

### Testing & Evaluation

| Command | Purpose |
|---------|---------|
| `pytest tests/ -v` | Run all unit tests |
| `pytest tests/test_embeddings.py -v` | Run embedding tests only |
| `jupyter notebook` | Open Jupyter to run notebooks |
| Open `notebooks/evaluate_recall_at_k.ipynb` | Evaluate retrieval quality (recall@k, MRR, etc.) |
| Open `notebooks/model_comparison.ipynb` | Compare embedding models |

### Infrastructure

| Command | Purpose |
|---------|---------|
| `cd semantic_image_search/infra && docker-compose up -d` | Start local Qdrant + services |
| `docker-compose down` | Stop services |
| `docker build -f Dockerfile -t semantic-image-search:latest .` | Build container image |

## Configuration

All configuration is managed through YAML files in `semantic_image_search/config/`:

- **`qdrant.yaml`** — Qdrant connection details (host, port, API key)
- **`s3.yaml`** — AWS S3 credentials (optional, for cloud storage)
- **`logging.yaml`** — Log level, format, output destination
- **`settings.yaml`** — Application settings (batch size, model name, etc.)

For sensitive data (API keys, passwords), use **environment variables**:
```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export S3_ACCESS_KEY=your_key
```

## Project Workflow

1. **Ingestion** → Load images from disk, generate captions/embeddings → Store in Qdrant
2. **Query** → Accept text or image query → Generate embedding → Search similar vectors → Rank results
3. **Evaluation** → Compute recall@k, MRR, NDCG metrics on test set
4. **Serving** → FastAPI backend exposes `/search`, `/ingest`, `/health` endpoints
5. **UI** → Streamlit app provides interactive search interface

## Key Modules

| Module | Purpose |
|--------|---------|
| `backend/embeddings.py` | Vision-language model integration, embedding generation |
| `backend/ingestion.py` | Image loading, batch processing, storage upsert |
| `backend/qdrant_client.py` | Wrapper around Qdrant Python client for search/index ops |
| `backend/retriever.py` | Query translation, vector search, result ranking |
| `src/api/` | FastAPI routes and request/response schemas |
| `src/models/` | Pydantic models for type safety |
| `src/storage/` | Abstract storage layer (local filesystem, S3) |
| `src/vectorstore/` | Vector DB interface and implementations |
| `ui/app.py` | Streamlit interactive UI |

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=semantic_image_search
```

Test categories:
- `test_api.py` — API endpoint tests
- `test_embeddings.py` — Embedding generation and model tests
- `test_ingestion.py` — Image loading and processing tests
- `test_vectorstore.py` — Qdrant integration tests
- `test_storage.py` — File and S3 storage tests

## Deployment

### Docker

```bash
# Build image
docker build -f semantic_image_search/infra/Dockerfile -t semantic-image-search:v1.0 .

# Run container
docker run -p 8000:8000 -e QDRANT_HOST=qdrant semantic-image-search:v1.0
```

### Kubernetes

Manifests available in `semantic_image_search/infra/k8s/`. Includes:
- Deployment for backend service
- Service exposing port 8000
- ConfigMap for YAML configs
- Secrets for API keys

```bash
kubectl apply -f semantic_image_search/infra/k8s/
```

### CI/CD

GitHub Actions workflows in `semantic_image_search/infra/github-actions/`:
- Run tests on push
- Build and push Docker images
- Deploy to staging/production

## Monitoring & Logging

- Health check: `GET /health`
- Metrics: Prometheus-compatible endpoints (if monitoring enabled)
- Logs: Configured via `config/logging.yaml` — output to console or file

## Performance Tips

1. **Batch ingestion** — Use batching in `ingestion.py` for large image sets
2. **Vector indexing** — Qdrant HNSW indexing is fast for similarity search (tuned in `qdrant.yaml`)
3. **Caching** — Consider caching embeddings for repeated queries
4. **Model selection** — Choose compact models for speed vs. larger models for quality

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Qdrant connection refused | Ensure Qdrant is running (`docker-compose up -d` or verify host/port in config) |
| Out of memory during ingestion | Reduce batch size in `settings.yaml` or `ingestion.py` |
| Missing embeddings | Check model is downloaded and available; see `embeddings.py` |
| Tests fail | Run `get_lib_versions.py` to verify all dependencies installed |

## Next Steps

- Fine-tune embedding model on domain-specific images
- Add user authentication and per-user query history
- Implement result caching layer (Redis)
- Expand UI with filters, facets, and saved searches
- Set up monitoring dashboard (Grafana)
- Add API rate limiting and request logging

