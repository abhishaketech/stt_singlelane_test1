from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
from datetime import datetime
import json
import asyncio
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

# Ensure the engine is available
try:
    from faster_whisper_engine import STTPipeline, ModelSize, ComputeType
except ImportError:
    raise ImportError("Could not import STT Pipeline. Ensure 'faster_whisper_engine.py' is in the same folder.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for cleaner production logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stt_api.log", encoding='utf-8')
    ]
)

logger = logging.getLogger("stt_api")

# --- FastAPI Setup ---
app = FastAPI(
    title="STT & Biometrics API",
    description="Real-time Speech-to-Text with Speaker Embeddings (Resemblyzer/GE2E)",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directories ---
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("transcription_results")
RESULTS_DIR.mkdir(exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# --- Global Resources ---
executor = ThreadPoolExecutor(max_workers=4)

# Initialize Pipelines
try:
    # 1. Initialize Whisper Pipeline (Now includes VoiceBiometricsEngine internally)
    pipeline = STTPipeline(
        model_size=ModelSize.BASE,
        device="auto",
        compute_type=ComputeType.FLOAT32,
        logfile=str(LOG_DIR / "pipeline.log"),
    )
    logger.info("STT Pipeline & Biometrics Engine initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize STT Pipeline: {str(e)}")
    raise


# --- Helper Functions ---

def sanitize_filename(filename: str) -> str:
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type not allowed: {ext}")

    safe_filename = f"{uuid.uuid4()}_{safe_name}"
    return safe_filename.replace("/", "_").replace("\\", "_")


# --- Event Handlers ---

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down STT Pipeline API")
    for file in TEMP_DIR.glob("*"):
        try:
            file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")


# --- REST Endpoints ---

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": pipeline.engine.get_model_info(),
        "version": "4.0.0",
        "features": ["vad_filter", "speaker_embedding_256d"]
    })


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    call_id: str = Query(None, description="Unique Session/Call ID for tagging embeddings"),
    language: str = Query(None, description="Language code (e.g., 'en', 'hi')"),
    word_level: bool = Query(False, description="Extract word-level timestamps"),
    vad_filter: bool = Query(True, description="Apply voice activity detection"),
):
    temp_path = None
    try:
        safe_filename = sanitize_filename(file.filename)
        temp_path = TEMP_DIR / safe_filename
        
        # Determine Call ID (Auto-generate if missing)
        final_call_id = call_id or f"rest-{uuid.uuid4()}"

        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        logger.info(f"Processing file: {file.filename} (Call ID: {final_call_id})")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            pipeline.process_audio,
            str(temp_path),
            language,
            None,   # sample_rate (auto-detect)
            word_level,
            vad_filter,
            final_call_id # <--- Passing Call ID to Engine
        )

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@app.post("/batch-transcribe")
async def batch_transcribe(
    files: list[UploadFile] = File(...),
    language: str = Query(None),
    word_level: bool = Query(False),
    vad_filter: bool = Query(True),
):
    temp_files = []
    results = []

    try:
        for file in files:
            safe_filename = sanitize_filename(file.filename)
            temp_path = TEMP_DIR / safe_filename
            content = await file.read()
            with open(temp_path, "wb") as buffer:
                buffer.write(content)
            temp_files.append((temp_path, file.filename))

        logger.info(f"Starting batch transcription for {len(temp_files)} files")

        loop = asyncio.get_event_loop()
        for temp_path, original_filename in temp_files:
            try:
                # Generate unique ID per file in batch
                file_call_id = f"batch-{uuid.uuid4()}"
                
                result = await loop.run_in_executor(
                    executor,
                    pipeline.process_audio,
                    str(temp_path),
                    language,
                    None,
                    word_level,
                    vad_filter,
                    file_call_id
                )
                result["filename"] = original_filename
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {original_filename}: {e}")
                results.append({"file": original_filename, "error": str(e)})

        return JSONResponse({
            "total_files": len(results),
            "results": results,
        })

    finally:
        for temp_path, _ in temp_files:
            if temp_path.exists():
                temp_path.unlink()


# --- WebSocket Streaming Logic ---

class MicrophoneStreamBuffer:
    def __init__(self, engine, call_id: str, language: str = None, word_level: bool = False, sample_rate: int = 16000):
        self.engine = engine
        self.call_id = call_id
        self.language = language
        self.word_level = word_level
        self.sample_rate = sample_rate
        self.audio_buffer = []
        self.total_samples = 0
        self.chunk_count = 0
        self.logger = logging.getLogger("microphone_stream")

    def add_audio_chunk(self, audio_chunk: np.ndarray) -> Dict:
        # We just buffer here. The sophisticated extraction happens in 'transcribe_stream'
        # at the end of the session to ensure we have the full context for stability.
        self.audio_buffer.append(audio_chunk)
        self.chunk_count += 1
        self.total_samples += len(audio_chunk)
        
        return {
            "status": "chunk_received",
            "chunk_number": self.chunk_count,
            "duration": round(self.total_samples / self.sample_rate, 2)
        }

    def transcribe_stream(self) -> Optional[Dict]:
        if not self.audio_buffer:
            return None

        try:
            full_audio = np.concatenate(self.audio_buffer)
            self.logger.info(f"Finalizing stream for Call ID: {self.call_id}. Audio duration: {len(full_audio)/self.sample_rate:.2f}s")
            
            # The Engine now handles VAD and Embedding Extraction internally
            result = self.engine.transcribe(
                full_audio,
                call_id=self.call_id,    # <--- Pass Call ID here
                language=self.language,
                sr=self.sample_rate,
                word_level=self.word_level,
                vad_filter=True,
            )

            return result.to_dict()

        except Exception as e:
            self.logger.error(f"Stream transcription failed: {str(e)}", exc_info=True)
            raise


@app.websocket("/ws/transcribe-stream")
async def websocket_transcribe_microphone(websocket: WebSocket):
    await websocket.accept()
    stream_buffer = None

    try:
        # Step 1: Receive Config Handshake
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)

        sample_rate = config.get("sample_rate", 16000)
        language = config.get("language")
        word_level = config.get("word_level", False)
        
        # Get Call ID from Client or Generate New
        call_id = config.get("call_id") or f"ws-{uuid.uuid4()}"

        # Initialize Buffer
        stream_buffer = MicrophoneStreamBuffer(
            pipeline.engine,
            call_id=call_id,
            language=language,
            word_level=word_level,
            sample_rate=sample_rate
        )

        logger.info(f"WebSocket started. Call ID: {call_id}")

        await websocket.send_json({
            "status": "ready",
            "call_id": call_id,
            "message": "Send audio chunks (float32 bytes)",
            "config": config
        })

        # Step 2: Receive Audio Stream
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=300.0)

                if "bytes" in data:
                    audio_bytes = data["bytes"]
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

                    loop = asyncio.get_event_loop()
                    status = await loop.run_in_executor(
                        executor,
                        stream_buffer.add_audio_chunk,
                        audio_chunk,
                    )
                    
                    # Optional: Send lightweight ack to keep connection alive
                    if status["chunk_number"] % 10 == 0:
                        await websocket.send_json(status)

                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "end":
                        break

            except asyncio.TimeoutError:
                logger.warning("WebSocket timeout")
                break

        # Step 3: Final Transcription & Embedding
        loop = asyncio.get_event_loop()
        final_result = await loop.run_in_executor(
            executor,
            stream_buffer.transcribe_stream,
        )

        # Build Response
        if final_result:
            # Auto-save Logic
            filename = f"recording_{final_result.get('call_id', 'unknown')}.json"
            save_path = RESULTS_DIR / filename
            
            response = {
                "status": "complete",
                "call_id": final_result.get("call_id"),
                "transcription": final_result.get("text", ""),
                "segments": final_result.get("segments", []),
                "voice_embedding_b64": final_result.get("voice_embedding_b64"), # The Base64 embedding
                "voice_embedding_exists": final_result.get("voice_embedding") is not None,
                "saved_file": str(filename)
            }
            
            with open(save_path, "w") as f:
                json.dump(final_result, f, indent=2) # Save full detail including segment timestamps
            
            await websocket.send_json(response)
        else:
            await websocket.send_json({"status": "error", "message": "No audio processed"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket critical error: {e}")
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except: pass
    finally:
        try: await websocket.close() 
        except: pass
@app.get("/")
async def root():
    return JSONResponse({
        "name": "STT & Biometrics API v4.0",
        "status": "Running",
        "endpoints": {
            "health_check": "/health",
            "documentation": "/docs",
            "websocket": "/ws/transcribe-stream"
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)