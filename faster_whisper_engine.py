import logging
import os
import time
import numpy as np
import librosa
import torch
import uuid
import base64
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Dependency Checks ---
try:
    from faster_whisper import WhisperModel
except ImportError as import_error:
    raise ImportError(
        "faster-whisper library not found. Install it with: pip install faster-whisper"
    ) from import_error

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("WARNING: 'resemblyzer' not found. Speaker embeddings will be disabled.")


# --- Enums & Data Structures ---

class ModelSize(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v2"
    LARGEV3 = "large-v3"

class ComputeType(str, Enum):
    INT8 = "int8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"

@dataclass
class TranscriptionWord:
    word: str
    confidence: float
    start: float
    end: float

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TranscriptionSegment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    confidence: float = None
    words: List[TranscriptionWord] = field(default_factory=list)

    def __post_init__(self):
        if self.confidence is None:
            self.confidence = max(0.0, 1.0 - self.no_speech_prob)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["words"] = [w.to_dict() for w in self.words]
        return data

@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: List[TranscriptionSegment]
    processing_time: float = None
    model_size: str = None
    device: str = None
    audio_file: str = None
    timestamp: str = None
    
    # --- New Speaker Embedding Fields ---
    call_id: str = None
    voice_embedding: List[float] = None       # 256-dim float array
    voice_embedding_b64: str = None           # Base64 encoded bytes for storage
    # ------------------------------------

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["segments"] = [s.to_dict() for s in self.segments]
        # Remove raw numpy arrays if any remain (safety)
        return {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}

    def to_json(self, output_file: Optional[str] = None) -> str:
        json_data = self.to_dict()
        json_str = json.dumps(json_data, indent=2)
        if output_file:
            Path(output_file).write_text(json_str)
        return json_str


# --- Utilities ---

class LoggerConfig:
    loggers = {}
    lock = Lock()

    @staticmethod
    def get_logger(name: str, logfile: Optional[str] = None) -> logging.Logger:
        with LoggerConfig.lock:
            if name in LoggerConfig.loggers:
                return LoggerConfig.loggers[name]
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
                if logfile:
                    file_handler = logging.FileHandler(logfile, encoding='utf-8')
                    file_handler.setLevel(logging.DEBUG)
                    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
            LoggerConfig.loggers[name] = logger
            return logger

class AudioProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.target_sr = 16000

    def load_audio(self, audio: Union[str, Path, np.ndarray], sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if isinstance(audio, np.ndarray):
            if sr is None: sr = 16000
            audio_float = audio.astype(np.float32)
            if sr != self.target_sr:
                audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=self.target_sr)
            return audio_float, self.target_sr
        
        # File loading
        y, sr = librosa.load(str(audio), sr=self.target_sr)
        return y.astype(np.float32), self.target_sr

    def detect_vad(self, audio: np.ndarray, sr: int, frame_length: int = 2048, energy_threshold: float = 0.02) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Simple energy-based VAD. Returns (original_audio, list_of_speech_segments).
        """
        hop_length = frame_length // 4
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
        energy = np.sqrt(np.mean(S ** 2, axis=0))
        energy_norm = energy / (np.max(energy) + 1e-8)
        
        active_frames = energy_norm > energy_threshold
        frames_time = librosa.frames_to_time(np.arange(len(energy_norm)), sr=sr, hop_length=hop_length)
        
        segments = []
        start_t = None
        for i, active in enumerate(active_frames):
            if active and start_t is None:
                start_t = frames_time[i]
            elif not active and start_t is not None:
                segments.append((float(start_t), float(frames_time[i])))
                start_t = None
        if start_t is not None:
            segments.append((float(start_t), float(frames_time[-1])))
            
        return audio, (segments if segments else [(0.0, len(audio)/sr)])


# --- 256-Dim Embedding Engine ---

class VoiceBiometricsEngine:
    def __init__(self, device: str = "cpu", logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("VoiceBiometrics")
        self.encoder = None
        
        if RESEMBLYZER_AVAILABLE:
            try:
                # Resemblyzer uses CPU efficiently; safer for avoiding VRAM clashes with Whisper
                self.encoder = VoiceEncoder(device="cpu", verbose=False) 
                self.logger.info("Voice Encoder (Resemblyzer/GE2E) initialized on CPU.")
            except Exception as e:
                self.logger.error(f"Failed to initialize VoiceEncoder: {e}")

    def extract_embedding(self, audio_segments_concat: np.ndarray, sr: int) -> Tuple[Optional[List[float]], Optional[str]]:
        """
        Generates one stable embedding from the concatenated speech segments.
        Returns: (List[float], Base64_String)
        """
        if self.encoder is None:
            return None, None
            
        # 1. Duration Check (Min 0.5s for stability)
        duration = len(audio_segments_concat) / sr
        if duration < 0.5:
            self.logger.warning(f"Speech duration too short for embedding ({duration:.2f}s). Skipping.")
            return None, None
            
        try:
            # 2. Preprocess (Ensure 16k, norm, trim)
            processed_wav = preprocess_wav(audio_segments_concat, source_sr=sr)
            
            # 3. Embed (Internally slices, embeds, and averages for stability)
            embedding_np = self.encoder.embed_utterance(processed_wav)
            
            # 4. Serialize
            embedding_list = embedding_np.tolist()
            b64_str = base64.b64encode(embedding_np.astype(np.float32).tobytes()).decode("utf-8")
            
            return embedding_list, b64_str
            
        except Exception as e:
            self.logger.error(f"Embedding extraction error: {e}")
            return None, None


# --- Main Engine ---

class FasterWhisperEngine:
    def __init__(self, model_size=ModelSize.BASE, device="auto", compute_type=ComputeType.FLOAT32, logfile=None, num_workers=1):
        self.logger = LoggerConfig.get_logger("FasterWhisperEngine", logfile=logfile)
        self.model_size = model_size
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type
        
        self.audio_processor = AudioProcessor(self.logger)
        self.biometrics = VoiceBiometricsEngine(logger=self.logger) # Init Biometrics
        
        self.logger.info(f"Loading Whisper Model: {model_size} on {self.device}...")
        try:
            self.model = WhisperModel(
                model_size.value, 
                device=self.device, 
                compute_type=self.compute_type.value, 
                num_workers=num_workers
            )
        except Exception as e:
            # Fallback for CPU float16 issue
            if "float16" in str(e) and self.device == "cpu":
                self.logger.warning("Falling back to float32 for CPU.")
                self.model = WhisperModel(model_size.value, device="cpu", compute_type="float32")
            else:
                raise

    def transcribe(self, audio, call_id=None, language=None, sr=None, word_level=False, vad_filter=True) -> TranscriptionResult:
        start_time = time.time()
        
        # 0. Ensure Call ID
        if not call_id:
            call_id = f"auto-{uuid.uuid4()}"
            
        # 1. Load Audio
        audio_data, sample_rate = self.audio_processor.load_audio(audio, sr)
        audio_duration = len(audio_data) / sample_rate
        
        # 2. VAD & Embedding
        vad_segments = [(0.0, audio_duration)]
        embedding_list = None
        embedding_b64 = None
        
        if vad_filter:
            audio_data, vad_segments = self.audio_processor.detect_vad(audio_data, sample_rate)
            
            # --- Biometric Extraction Step ---
            # We reconstruct "pure speech" by joining all VAD segments
            speech_chunks = []
            for start, end in vad_segments:
                s = int(start * sample_rate)
                e = int(end * sample_rate)
                speech_chunks.append(audio_data[s:e])
            
            if speech_chunks:
                pure_speech = np.concatenate(speech_chunks)
                # Generate stable embedding from pure speech
                embedding_list, embedding_b64 = self.biometrics.extract_embedding(pure_speech, sample_rate)
                if embedding_list:
                    self.logger.info(f"Stable embedding generated for Call ID: {call_id}")
            # ---------------------------------

        # 3. Transcribe
        segments_gen, detected_info = self.model.transcribe(
            audio_data, language=language, word_timestamps=word_level, beam_size=5
        )
        
        # Convert generator to list
        segments = []
        for i, s in enumerate(segments_gen):
            # Map faster_whisper segment to our dataclass
            words = [TranscriptionWord(w.word, w.probability, w.start, w.end) for w in (s.words or [])]
            segments.append(TranscriptionSegment(
                id=i, seek=s.seek, start=s.start, end=s.end, text=s.text.strip(),
                tokens=s.tokens, temperature=s.temperature, avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio, no_speech_prob=s.no_speech_prob, words=words
            ))

        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            text=" ".join([s.text for s in segments]),
            language=detected_info.language,
            duration=audio_duration,
            segments=segments,
            processing_time=processing_time,
            model_size=self.model_size.value,
            device=self.device,
            
            # Tags
            call_id=call_id,
            voice_embedding=embedding_list,
            voice_embedding_b64=embedding_b64
        )

    def __del__(self):
        if hasattr(self, "model"): del self.model
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# --- Pipeline Wrapper ---

class STTPipeline:
    def __init__(self, model_size=ModelSize.BASE, device="auto", compute_type=ComputeType.FLOAT32, logfile=None):
        self.engine = FasterWhisperEngine(model_size, device, compute_type, logfile)

    def process_audio(self, audio, language=None, sr=None, word_level=False, vad_filter=True, call_id=None):
        return self.engine.transcribe(
            audio, call_id=call_id, language=language, sr=sr, word_level=word_level, vad_filter=vad_filter
        ).to_dict()