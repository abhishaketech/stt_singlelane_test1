from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
from datetime import datetime
import json
import asyncio
import uuid
import base64
import numpy as np
import noisereduce as nr  # <--- Required for Denoising
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

# Ensure the engine is available
try:
    from faster_whisper_engine import STTPipeline, ModelSize, ComputeType
except ImportError:
    raise ImportError("Could not import STT Pipeline. Ensure 'faster_whisper_engine.py' is in the same folder.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stt_api.log", encoding='utf-8')
    ]
)

logger = logging.getLogger("stt_api")

# --- FastAPI Setup ---
app = FastAPI(
    title="STT Microphone Streaming API",
    description="Real-time Speech-to-Text with microphone streaming, batch processing, and Aggressive Denoising",
    version="3.3.1 (Aggressive Denoising)",
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
    # 1. Initialize Whisper Pipeline
    pipeline = STTPipeline(
        model_size=ModelSize.BASE,
        device="auto",
        compute_type=ComputeType.FLOAT32,
        logfile=str(LOG_DIR / "pipeline.log"),
    )
    logger.info("STT Pipeline initialized successfully")

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

    if len(safe_name) > 255:
        name, ext = Path(safe_name).stem, Path(safe_name).suffix
        safe_name = name[:200] + ext

    safe_filename = f"{uuid.uuid4()}_{safe_name}"
    safe_filename = safe_filename.replace("/", "_").replace("\\", "_")

    return safe_filename

def extract_speaker_embedding(waveform: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Returns the raw waveform (numpy array) to act as the Audio Fingerprint.
    """
    try:
        logger.debug(f"Extracting raw audio fingerprint: {len(waveform)} samples")
        return waveform
    except Exception as e:
        logger.error(f"Fingerprint extraction failed: {e}")
        return None

def serialize_voice_fingerprint(fingerprint_sample: Optional[np.ndarray]) -> Optional[Dict]:
    """
    Converts the fingerprint (raw audio data) into a Base64 string.
    """
    if fingerprint_sample is None:
        return None

    try:
        # Convert the float32 numpy array to raw bytes
        fp_bytes = fingerprint_sample.astype(np.float32).tobytes()
        
        # Encode bytes to Base64 String
        b64_string = base64.b64encode(fp_bytes).decode('utf-8')
        
        return {
            "voice_fingerprint_sample_b64": b64_string,
            "voice_fingerprint_sample_shape": list(fingerprint_sample.shape),
            "voice_fingerprint_sample_dtype": "float32 (raw audio)",
            "voice_fingerprint_type": "raw_base64"
        }
    except Exception as e:
        logger.warning(f"Failed to serialize voice fingerprint: {e}")
        return None


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
        "version": "3.3.1",
        "features": ["base64_raw_fingerprint", "aggressive_noise_reduction"]
    })


@app.get("/model-info")
async def model_info():
    return JSONResponse(pipeline.engine.get_model_info())


@app.get("/supported-languages")
async def supported_languages():
    return JSONResponse(pipeline.engine.get_supported_languages())


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query(None, description="Language code (e.g., 'en', 'hi', 'zh')"),
    word_level: bool = Query(False, description="Extract word-level timestamps"),
    vad_filter: bool = Query(True, description="Apply voice activity detection"),
):
    temp_path = None
    try:
        safe_filename = sanitize_filename(file.filename)
        temp_path = TEMP_DIR / safe_filename

        if not temp_path.resolve().is_relative_to(TEMP_DIR.resolve()):
            raise ValueError("Invalid file path")

        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        logger.info(f"Processing file: {file.filename}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            pipeline.process_audio,
            str(temp_path),
            language,
            None,
            word_level,
            vad_filter,
        )

        logger.info(f"Transcription successful for {file.filename}")
        return JSONResponse(result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.post("/batch-transcribe")
async def batch_transcribe(
    files: list[UploadFile] = File(...),
    language: str = Query(None, description="Language code for all files"),
    word_level: bool = Query(False),
    vad_filter: bool = Query(True),
):
    temp_files = []
    results = []

    try:
        for file in files:
            try:
                safe_filename = sanitize_filename(file.filename)
                temp_path = TEMP_DIR / safe_filename
                content = await file.read()

                with open(temp_path, "wb") as buffer:
                    buffer.write(content)

                temp_files.append((temp_path, file.filename))
            except Exception as e:
                logger.error(f"Failed to save file {file.filename}: {e}")
                results.append({
                    "file": file.filename,
                    "status": "failed",
                    "error": f"File save error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                })

        logger.info(f"Starting batch transcription for {len(temp_files)} files")

        loop = asyncio.get_event_loop()
        for temp_path, original_filename in temp_files:
            try:
                result = await loop.run_in_executor(
                    executor,
                    pipeline.process_audio,
                    str(temp_path),
                    language,
                    None,
                    word_level,
                    vad_filter,
                )

                result["filename"] = original_filename
                results.append(result)
                logger.info(f"Processed: {original_filename}")
            except Exception as e:
                logger.error(f"Failed to process {original_filename}: {str(e)}")
                results.append({
                    "file": original_filename,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

        logger.info(f"Batch transcription completed: {len(results)} results")
        return JSONResponse({
            "total_files": len(results),
            "timestamp": datetime.now().isoformat(),
            "results": results,
        })

    except Exception as e:
        logger.error(f"Batch transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for temp_path, _ in temp_files:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")


# --- WebSocket Streaming Logic (MODIFIED FOR AGGRESSIVE DENOISING) ---

class MicrophoneStreamBuffer:
    def __init__(self, engine, language: str = None, word_level: bool = False, sample_rate: int = 16000):
        self.engine = engine
        self.language = language
        self.word_level = word_level
        self.sample_rate = sample_rate
        self.audio_buffer = []
        self.total_samples = 0
        self.fingerprint_duration_samples = int(6.0 * sample_rate)
        self.fingerprint_extracted = False
        self.voice_fingerprint = None
        self.chunk_count = 0
        self.logger = logging.getLogger("microphone_stream")

    def add_audio_chunk(self, audio_chunk: np.ndarray) -> Dict:
        self.audio_buffer.append(audio_chunk)
        self.chunk_count += 1
        self.total_samples += len(audio_chunk)

        # Attempt extraction exactly when we cross the 6-second threshold
        if not self.fingerprint_extracted and self.total_samples >= self.fingerprint_duration_samples:
            try:
                full_audio = np.concatenate(self.audio_buffer)
                six_sec_audio = full_audio[:self.fingerprint_duration_samples]
                
                # --- AGGRESSIVE DENOISING STEP ---
                try:
                    self.logger.info("Applying AGGRESSIVE noise reduction to fingerprint...")
                    # Perform stationary noise reduction with aggressive params
                    clean_audio = nr.reduce_noise(
                        y=six_sec_audio, 
                        sr=self.sample_rate, 
                        stationary=True,         # Assumes fan/AC/hiss is constant
                        prop_decrease=1.0,       # Reduce noise by 100%
                        n_std_thresh_stationary=1.5, # Aggressive threshold
                        n_fft=2048               # Higher resolution FFT
                    )
                except Exception as noise_err:
                    self.logger.error(f"Denoising failed (using raw audio instead): {noise_err}")
                    clean_audio = six_sec_audio
                # ---------------------------------

                # Extract fingerprint from CLEAN audio
                self.voice_fingerprint = extract_speaker_embedding(
                    clean_audio,
                    self.sample_rate
                )
                
                if self.voice_fingerprint is not None:
                    self.fingerprint_extracted = True
                    self.logger.info(f"Raw Audio captured & cleaned (Samples: {self.voice_fingerprint.shape[0]})")
                else:
                    self.logger.warning("Voice fingerprint extraction returned None")
            except Exception as e:
                self.logger.error(f"Error during fingerprint extraction: {e}")

        current_duration = self.total_samples / self.sample_rate

        return {
            "status": "chunk_received",
            "chunk_number": self.chunk_count,
            "total_duration_seconds": round(current_duration, 2),
            "fingerprint_extracted": self.fingerprint_extracted,
            "fingerprint_ready": self.fingerprint_extracted,
        }

    def get_full_audio(self) -> Optional[np.ndarray]:
        if not self.audio_buffer:
            return None
        return np.concatenate(self.audio_buffer)

    def transcribe_stream(self) -> Optional[Dict]:
        if not self.audio_buffer:
            self.logger.warning("No audio data in buffer")
            return None

        try:
            full_audio = self.get_full_audio()
            # Call the engine's transcribe method directly
            result = self.engine.transcribe(
                full_audio,
                language=self.language,
                sr=self.sample_rate,
                word_level=self.word_level,
                vad_filter=True,
            )

            # Attach the fingerprint if we captured one during the stream
            if self.voice_fingerprint is not None:
                result.voice_fingerprint_sample = self.voice_fingerprint

            self.logger.info(f"Stream transcription complete: {len(full_audio)} samples processed")
            return result.to_dict()

        except Exception as e:
            self.logger.error(f"Stream transcription failed: {str(e)}", exc_info=True)
            raise


@app.websocket("/ws/transcribe-stream")
async def websocket_transcribe_microphone(websocket: WebSocket):
    await websocket.accept()
    stream_buffer = None

    try:
        # Step 1: Receive Config
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)

        sample_rate = config.get("sample_rate", 16000)
        language = config.get("language")
        word_level = config.get("word_level", False)

        # Initialize Buffer
        stream_buffer = MicrophoneStreamBuffer(
            pipeline.engine,
            language=language,
            word_level=word_level,
            sample_rate=sample_rate
        )

        logger.info(f"Microphone stream started: sr={sample_rate}Hz, lang={language}")

        await websocket.send_json({
            "status": "ready",
            "message": "Ready to receive microphone audio chunks",
            "sample_rate": sample_rate,
            "language": language,
        })

        # Step 2: Receive Audio Stream
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=300.0)

                if "bytes" in data:
                    audio_bytes = data["bytes"]
                    # Assume Float32 input from client
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

                    loop = asyncio.get_event_loop()
                    chunk_status = await loop.run_in_executor(
                        executor,
                        stream_buffer.add_audio_chunk,
                        audio_chunk,
                    )

                    await websocket.send_json({
                        "status": "chunk_processed",
                        "chunk_number": chunk_status["chunk_number"],
                        "total_duration_seconds": chunk_status["total_duration_seconds"],
                        "fingerprint_extracted": chunk_status["fingerprint_extracted"],
                    })

                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "end":
                        logger.info(f"Stream end signal received after {stream_buffer.chunk_count} chunks")
                        break

            except asyncio.TimeoutError:
                logger.warning("WebSocket timeout - stream inactive for 5 minutes")
                break

        # Step 3: Final Transcription
        loop = asyncio.get_event_loop()
        final_result = await loop.run_in_executor(
            executor,
            stream_buffer.transcribe_stream,
        )

        # Build response
        if final_result:
            
            # 1. Create the Base Response First
            response = {
                "status": "complete",
                "transcription": final_result.get("text", ""),
                "segments": final_result.get("segments", []),
                "chunks_processed": stream_buffer.chunk_count,
                "total_duration_seconds": round(stream_buffer.total_samples / stream_buffer.sample_rate, 2),
                "fingerprint_extracted": stream_buffer.fingerprint_extracted,
            }

            # 2. Add the Fingerprint (Base64) to the response
            if stream_buffer.voice_fingerprint is not None:
                fp = serialize_voice_fingerprint(stream_buffer.voice_fingerprint)
                if fp:
                    response.update(fp)

            # 3. --- AUTO-SAVE LOGIC ---
            filename = f"recording_{uuid.uuid4()}.json"
            save_path = RESULTS_DIR / filename
            
            # Add filename to response so frontend knows what it was saved as
            response["saved_file"] = str(filename)
            
            try:
                with open(save_path, "w") as f:
                    # Save the FULL response (which now includes Base64 & Cleaned Audio)
                    json.dump(response, f, indent=2) 
                logger.info(f"Saved recording result to {save_path}")
            except Exception as e:
                logger.error(f"Failed to auto-save result: {e}")
            # ------------------------------------------------

            await websocket.send_json(response)

        else:
            await websocket.send_json({
                "status": "complete",
                "chunks_processed": stream_buffer.chunk_count,
                "warning": "No audio data received",
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "status": "error",
                "error": str(e),
            })
        except:
            pass

    finally:
        try:
            await websocket.close(code=1000)
        except:
            pass

# --- Result Management ---

@app.get("/results")
async def list_results(limit: int = Query(10, ge=1, le=100)):
    try:
        results = sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:limit]
        result_list = []

        for result_file in results:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    result_list.append({
                        "file": result_file.name,
                        "timestamp": result_file.stat().st_mtime,
                        "text_preview": data.get("text", "")[:100],
                    })
            except Exception as e:
                logger.warning(f"Failed to read result file: {e}")

        return JSONResponse({
            "total": len(result_list),
            "results": result_list,
        })

    except Exception as e:
        logger.error(f"Error listing results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{filename}")
async def get_result(filename: str):
    try:
        result_path = RESULTS_DIR / filename
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Result not found")

        return FileResponse(
            path=result_path,
            media_type="application/json",
            filename=filename,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/results/{filename}")
async def delete_result(filename: str):
    try:
        result_path = RESULTS_DIR / filename
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Result not found")

        result_path.unlink()
        logger.info(f"Deleted result file: {filename}")

        return JSONResponse({
            "status": "deleted",
            "file": filename,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def statistics():
    try:
        result_files = list(RESULTS_DIR.glob("*.json"))
        total_files = len(result_files)
        total_duration = 0
        total_processing_time = 0
        language_counts = {}

        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    total_duration += data.get("duration", 0)
                    total_processing_time += data.get("processing_time", 0)
                    lang = data.get("language", "unknown")
                    language_counts[lang] = language_counts.get(lang, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to parse result file: {e}")

        avg_rtf = (total_processing_time / total_duration) if total_duration > 0 else 0

        return JSONResponse({
            "total_transcriptions": total_files,
            "total_audio_duration": round(total_duration, 2),
            "total_processing_time": round(total_processing_time, 2),
            "average_real_time_factor": round(avg_rtf, 2),
            "languages_processed": language_counts,
            "model_info": pipeline.engine.get_model_info(),
        })

    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return JSONResponse({
        "name": "STT Microphone Streaming API v3.3.1 (Aggressive Denoising)",
        "version": "3.3.1",
        "description": "Real-time Speech-to-Text with microphone streaming, batch processing and Aggressive Denoising",
        "endpoints": {
            "POST /transcribe": "Transcribe single audio file",
            "POST /batch-transcribe": "Transcribe multiple audio files",
            "GET /health": "Health check endpoint",
            "WS /ws/transcribe-stream": "Real-time microphone audio streaming",
        },
        "timestamp": datetime.now().isoformat(),
    })


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting STT Microphone Streaming API v3.3.1")
    logger.info("Features: Real-time streaming, Raw Base64 Fingerprinting, Auto-Save, Aggressive Denoising")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        workers=1,
    )