# Symantec Search AI Embeddings Service

<img width="681" height="509" alt="image" src="https://github.com/user-attachments/assets/e183841a-93ea-4417-9421-5762b72dcaf5" />

<img width="709" height="558" alt="image" src="https://github.com/user-attachments/assets/96c9d81a-5042-49f2-a2ee-550941ada571" />

https://github.com/user-attachments/assets/e3a70de8-8f8f-4685-995e-98704aa9001f

A high-performance, local AI service providing **Speech-to-Text (STT)**, **Semantic Embeddings**, and **Hybrid Search scoring**. This service provides the **"Intelligence Layer"** for the Receipt Insights platform and is optimized for containerized environments with built-in performance monitoring.

---

## 🚀 Key Features

- **Semantic Search**: Powered by `sentence-transformers/all-MiniLM-L6-v2` for fast, accurate text-to-vector conversion (384-d).
- **Speech-to-Text**: Utilizes OpenAI Whisper (`base.en`) with merchant-specific context prompts for high-accuracy receipt transcription.
- **Hybrid Search (BM25)**: Support for traditional keyword relevance scoring via `rank-bm25` to complement vector similarity.
- **Sentiment Analysis**: Real-time "Vibe Gatekeeper" to moderate and flag unprofessional or negative input during ingestion.
- **Performance Monitoring**: Detailed per-request timing logs for file I/O, transcription, and embedding generation.
- **Health Monitoring**: Lightweight `/health` endpoint for Azure startup and liveness probes.

## 🛠 Tech Stack

- **API Framework**: FastAPI
- **NLP Models**: 
  - HuggingFace Transformers
  - Sentence Transformers
  - OpenAI Whisper
- **Ranking**: BM25 (Okapi)
- **Infrastructure**: Docker (Optimized for model caching)

## 📦 Deployment

### Docker (Recommended)

The service is designed to be managed by the root `docker-compose.yml`:

```yaml
services:
  embeddings-api:
    build:
      context: ./embeddings-api
    container_name: receipt-insights-embeddings-api
    ports:
      - "8000:8000"
    environment:
      - HF_HOME=/app/models
```

### Local Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness & Readiness check |
| `POST` | `/transcribe` | Converts audio files to text |
| `POST` | `/embed` | Generates semantic vectors from text |
| `POST` | `/bm25-score` | Calculates BM25 scores for hybrid search |
| `POST` | `/voice-search` | Combined STT + Embedding pipeline |
| `POST` | `/process-note` | Sentiment analysis + Embedding (Ingestion) |

---
*Maintained by the Symantec Search AI Team.*
