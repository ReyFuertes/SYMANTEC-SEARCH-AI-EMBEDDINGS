from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
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

# --- MODEL INITIALIZATION ---

# 1. Sentence Transformers: Converts text into 384-dimensional vectors.
# Model 'all-MiniLM-L6-v2' is chosen for its high speed and low memory footprint while maintaining excellent semantic accuracy.
# These vectors are used for Vector Similarity Search (VSS) in Redis Stack and pgvector.
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

# 2. OpenAI Whisper: High-accuracy Speech-to-Text (STT) model.
# Using the 'base' model for a balanced performance/accuracy ratio on CPU.
# This powers the voice notes on receipt uploads and natural language voice search queries.
model_stt = whisper.load_model("base")

# 3. Sentiment Analysis (Vibe Check): Detects the emotional tone of user input.
# Model 'distilbert-base-uncased-finetuned-sst-2-english' is a distilled version of BERT, 
# making it extremely fast for real-time moderation during ingestion.
vibe_checker = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- DATA MODELS ---

class TextRequest(BaseModel):
    text: str

# --- API ENDPOINTS ---

@app.post("/transcribe", summary="Speech-to-Text Transcription")
async def transcribe_audio(file: UploadFile):
    """
    Transcribes an uploaded audio file into text using OpenAI Whisper.
    Accepts common audio formats (WAV, MP3, AAC, etc.) via Multipart form data.
    """
    # Create a temporary file to store the uploaded audio for Whisper processing
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_name = tmp.name
    
    try:
        # Perform transcription
        result = model_stt.transcribe(tmp_name)
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
    vector = model_embed.encode(req.text).tolist()
    return {"vector": vector}

@app.post("/voice-search", summary="End-to-End Voice Search Pipeline")
async def voice_search(file: UploadFile):
    """
    A combined pipeline that:
    1. Transcribes audio input into text.
    2. Converts that text into a vector for instant searching.
    Primarily used for the 'Smart Assistant' search feature in the mobile app.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_name = tmp.name
        
    try:
        # Step 1: Speech -> Text
        transcription = model_stt.transcribe(tmp_name)["text"].strip()
        # Step 2: Text -> Semantic Vector
        vector = model_embed.encode(transcription).tolist()
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
