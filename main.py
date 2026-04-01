from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from typing import List
import whisper
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import tempfile
import os

# Initialize FastAPI App
app = FastAPI(
    title="Receipt Insights AI Service",
    description="Local AI Service for Speech-to-Text, Semantic Embeddings, and Sentiment Analysis",
    version="1.0.0"
)

import time

# --- MODEL INITIALIZATION ---
import datetime

def log_milestone(message):
    print(f"[{datetime.datetime.now().isoformat()}] [BOOTSTRAP] {message}", flush=True)

def load_models():
    try:
        log_milestone("Process started. Beginning model initialization...")
        
        log_milestone("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
        embed = SentenceTransformer('all-MiniLM-L6-v2')
        log_milestone("Sentence Transformer loaded successfully.")
        
        # Switched to base.en - much more accurate for merchant names while still fast on CPU
        log_milestone("Loading Whisper STT model (base.en)...")
        stt = whisper.load_model("base.en")
        log_milestone("Whisper STT loaded successfully.")
        
        log_milestone("Loading Sentiment Analysis model...")
        vibe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        log_milestone("Sentiment Analysis loaded successfully.")
        
        log_milestone("ALL MODELS LOADED. Web server starting...")
        return embed, stt, vibe
    except Exception as e:
        print(f"[CRITICAL ERROR] [{datetime.datetime.now().isoformat()}] Failed to load models: {str(e)}", flush=True)
        import sys
        sys.exit(1)

model_embed, model_stt, vibe_checker = load_models()

# --- HELPER CONSTANTS ---
# Providing a prompt helps Whisper recognize specific context and technical terms
STT_PROMPT = "Searching for my receipts: Jollibee, McDonald's, Starbucks, SM Store, Grab, Foodpanda, Grocery, Restaurant."

# --- DATA MODELS ---

class TextRequest(BaseModel):
    text: str

# --- API ENDPOINTS ---

@app.get("/health", summary="Health Check")
async def health_check():
    """
    Lightweight endpoint for Azure Startup and Liveness probes.
    """
    print("[PROBE] Health check requested")
    return {"status": "healthy"}

@app.post("/transcribe", summary="Speech-to-Text Transcription")
async def transcribe_audio(file: UploadFile):
    """
    Transcribes an uploaded audio file into text using OpenAI Whisper.
    Accepts common audio formats (WAV, MP3, AAC, etc.) via Multipart form data.
    """
    start_time = time.time()
    # Create a temporary file to store the uploaded audio for Whisper processing
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_name = tmp.name
    
    file_save_time = time.time() - start_time
    print(f"[PERF] File Save: {file_save_time:.2f}s", flush=True)

    try:
        # Perform transcription with accuracy optimizations
        transcribe_start = time.time()
        result = model_stt.transcribe(
            tmp_name, 
            fp16=False, 
            language="en", 
            initial_prompt=STT_PROMPT
        )
        transcribe_time = time.time() - transcribe_start
        print(f"[PERF] Whisper Transcribe: {transcribe_time:.2f}s", flush=True)
        
        return {"text": result["text"].strip()}
    finally:
        # Ensure temporary file is cleaned up after processing
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

@app.post("/embed", summary="Generate Semantic Embedding")
async def embed_text(req: TextRequest):
    """
    Converts a plain text string into a 384-dimensional floating-point vector.
    This vector represents the 'semantic meaning' of the text for similarity searches.
    """
    start_time = time.time()
    vector = model_embed.encode(req.text).tolist()
    duration = time.time() - start_time
    print(f"[PERF] Embedding Generation: {duration:.2f}s", flush=True)
    return {"vector": vector}

@app.post("/voice-search", summary="End-to-End Voice Search Pipeline")
async def voice_search(file: UploadFile):
    """
    A combined pipeline that:
    1. Transcribes audio input into text.
    2. Converts that text into a vector for instant searching.
    Primarily used for the 'Smart Assistant' search feature in the mobile app.
    """
    start_overall = time.time()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_name = tmp.name
        
    file_save_time = time.time() - start_overall
    print(f"[PERF] Voice Search - File Save: {file_save_time:.2f}s", flush=True)

    try:
        # Step 1: Speech -> Text (with optimizations)
        stt_start = time.time()
        transcription_result = model_stt.transcribe(
            tmp_name, 
            fp16=False, 
            language="en", 
            initial_prompt=STT_PROMPT
        )
        transcription = transcription_result["text"].strip()
        stt_time = time.time() - stt_start
        print(f"[PERF] Voice Search - STT: {stt_time:.2f}s", flush=True)

        # Step 2: Text -> Semantic Vector
        embed_start = time.time()
        vector = model_embed.encode(transcription).tolist()
        embed_time = time.time() - embed_start
        print(f"[PERF] Voice Search - Embed: {embed_time:.2f}s", flush=True)

        total_time = time.time() - start_overall
        print(f"[PERF] Voice Search - TOTAL: {total_time:.2f}s", flush=True)

        return {
            "query_text": transcription,
            "vector": vector
        }
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

@app.post("/process-note", summary="Contextual Ingestion & Moderation")
async def process_note(req: TextRequest):
    """
    The 'Vibe Gatekeeper':
    - Evaluates the sentiment of a user's voice note/description.
    - Flags unprofessional or negative tones (score > 0.95 negativity).
    - If passed, generates the embedding vector for database storage.
    """
    # Step 1: Perform Sentiment Analysis
    vibe = vibe_checker(req.text)[0]
    
    # Logic: If the model is 95%+ confident the tone is negative, we flag it for professionalism.
    if vibe['label'] == 'NEGATIVE' and vibe['score'] > 0.95:
        return {
            "status": "flagged",
            "reason": "Negative/Unprofessional Vibe detected.",
            "score": float(vibe['score'])
        }
    
    # Step 2: Generate Embedding for storage
    vector = model_embed.encode(req.text).tolist()
    return {
        "status": "success", 
        "vector": vector, 
        "score": float(vibe['score']),
        "label": vibe['label']
    }
