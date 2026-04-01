FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for whisper
RUN apt-get update && apt-get install -y ffmpeg wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a dedicated directory for models to avoid cache issues
RUN mkdir -p /app/models
ENV HF_HOME=/app/models
ENV XDG_CACHE_HOME=/app/models

# Pre-download and verify models
# 1. Sentence Transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# 2. Sentiment Analysis
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"
# 3. Whisper (Explicitly download to ensure it's not corrupted)
RUN python -c "import whisper; whisper.load_model('base.en')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]