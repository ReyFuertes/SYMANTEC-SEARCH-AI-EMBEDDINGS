# Receipt Insights: Embeddings & AI API

This service provides the **"Intelligence Layer"** for the Receipt Insights platform. It is a Python/FastAPI container hosting local open-source models for **Speech-to-Text (STT)**, **Semantic Embeddings**, and **Sentiment Analysis (Vibe Check)**.

---

## 🚀 Key Features

- **Whisper (Base)**: High-accuracy Speech-to-Text for receipt voice notes and voice-search queries.
- **all-MiniLM-L6-v2**: Converts receipt data into 384-dimensional vectors for ultra-fast **Semantic Search**.
- **distilbert-base-uncased-finetuned-sst-2-english**: Analyzes the "vibe" of user notes to ensure professionalism and compliance.
- **Local Hosting**: 100% free operation with total data privacy (no data leaves your Docker network).

---

## 🛠 API Endpoints

### 1. `POST /transcribe`
Converts an uploaded audio file into text.
- **Input**: `Multipart/form-data` (Audio file)
- **Output**: `{"text": "Lunch with the design team in Makati"}`

### 2. `POST /embed`
Converts a string into a 384-dimensional vector.
- **Input**: `{"text": "Starbucks Makati Coffee"}`
- **Output**: `{"vector": [0.12, -0.05, 0.88, ...]}`

### 3. `POST /process-note` (The "Vibe Gatekeeper")
Performs sentiment analysis and embedding in a single call.
- **Input**: `{"text": "This was a great business meeting!"}`
- **Output**: `{"status": "success", "vector": [...], "score": 0.99}`
- **Note**: Will return a `flagged` status if the tone is unprofessional or toxic.

### 4. `POST /voice-search`
The core of the Voice Search feature. Transcribes audio and immediately returns the vector for Redis search.
- **Input**: `Multipart/form-data` (Voice query)
- **Output**: `{"query_text": "Meetings in Makati", "vector": [...]}`

---

## 📦 Docker Integration

This service is managed by the root `docker-compose.yml`:

```yaml
  embeddings-api:
    build:
      context: ./embeddings-api
    container_name: receipt-insights-embeddings-api
    ports:
      - "8000:8000"
```

The Docker image pre-downloads all AI models during the build phase to ensure fast startup and offline readiness.

---

## 🐍 Local Development (Python)

To run this service outside of Docker for development:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # Ensure ffmpeg is installed on your system for Whisper
   ```

2. **Run the Server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access Documentation**:
   Navigate to `http://localhost:8000/docs` to see the interactive Swagger UI.

---

## 📝 Configuration

- **STT Model**: Currently using `base` for a balance of speed and accuracy. Update `main.py` to use `tiny` for speed or `small` for higher accuracy.
- **Embedding Model**: Using `all-MiniLM-L6-v2` which produces **384-dimensional** vectors. Ensure your `pgvector` column and Redis HNSW index are configured for this dimension.
