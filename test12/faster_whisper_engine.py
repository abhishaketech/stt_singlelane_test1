import logging
import os
import time
import numpy as np
import librosa
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from faster_whisper import WhisperModel
except ImportError as import_error:
    raise ImportError(
        "faster-whisper library not found. Install it with: pip install faster-whisper"
    ) from import_error


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
    voice_fingerprint_sample: np.ndarray = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["segments"] = [s.to_dict() for s in self.segments]
        if self.voice_fingerprint_sample is not None:
            data["voice_fingerprint_sample_shape"] = list(self.voice_fingerprint_sample.shape)
        return {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}

    def to_json(self, output_file: Optional[str] = None) -> str:
        json_data = self.to_dict()
        json_str = json.dumps(json_data, indent=2)
        if output_file:
            Path(output_file).write_text(json_str)
        return json_str

    def get_summary(self) -> Dict:
        avg_confidence = 0
        if self.segments:
            avg_confidence = sum(seg.confidence for seg in self.segments) / len(self.segments)

        total_words = 0
        for seg in self.segments:
            if hasattr(seg, 'words') and seg.words:
                total_words += len(seg.words)

        return {
            "text": self.text,
            "language": self.language,
            "duration": round(self.duration, 2),
            "processing_time": round(self.processing_time, 2),
            "real_time_factor": round(self.processing_time / self.duration, 2) if self.duration > 0 else 0,
            "segments_count": len(self.segments),
            "words_count": total_words,
            "avg_confidence": round(avg_confidence, 4),
            "device": self.device,
            "model_size": self.model_size,
            "voice_fingerprint_extracted": self.voice_fingerprint_sample is not None,
        }


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
                console_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)

                if logfile:
                    try:
                        log_path = Path(logfile)
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        file_handler = logging.FileHandler(logfile, encoding='utf-8')
                        file_handler.setLevel(logging.DEBUG)
                        file_formatter = logging.Formatter(
                            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S"
                        )
                        file_handler.setFormatter(file_formatter)
                        logger.addHandler(file_handler)
                    except Exception as log_error:
                        logger.warning(f"Could not create log file: {str(log_error)}")

            LoggerConfig.loggers[name] = logger
            return logger


class AudioValidator:
    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    SUPPORTED_SAMPLE_RATES = {8000, 16000, 22050, 44100, 48000}
    MIN_AUDIO_DURATION = 0.1
    MAX_AUDIO_DURATION = 3600

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate_file(self, audio_file: Union[str, Path]) -> Tuple[bool, str]:
        audio_path = Path(audio_file)
        if not audio_path.exists():
            return False, f"File does not exist: {audio_path}"

        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {audio_path.suffix}. Supported: {self.SUPPORTED_FORMATS}"

        if audio_path.stat().st_size == 0:
            return False, "Audio file is empty"

        try:
            duration = librosa.get_duration(filename=str(audio_path))
            if duration < self.MIN_AUDIO_DURATION:
                return False, f"Audio duration too short: {duration:.2f}s (minimum: {self.MIN_AUDIO_DURATION}s)"

            if duration > self.MAX_AUDIO_DURATION:
                return False, f"Audio duration too long: {duration:.2f}s (maximum: {self.MAX_AUDIO_DURATION}s)"

            return True, "Validation passed"
        except Exception as validation_error:
            return False, f"Error reading audio file: {str(validation_error)}"

    def validate_array(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, str]:
        if not isinstance(audio, np.ndarray):
            return False, "Audio must be a numpy array"

        if audio.size == 0:
            return False, "Audio array is empty"

        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            return False, f"Unsupported sample rate: {sample_rate}. Supported: {self.SUPPORTED_SAMPLE_RATES}"

        duration = len(audio) / sample_rate
        if duration < self.MIN_AUDIO_DURATION:
            return False, f"Audio duration too short: {duration:.2f}s"

        if duration > self.MAX_AUDIO_DURATION:
            return False, f"Audio duration too long: {duration:.2f}s"

        return True, "Validation passed"


class AudioProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.target_sr = 16000

    def load_audio(self, audio: Union[str, Path, np.ndarray], sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if isinstance(audio, np.ndarray):
            if sr is None:
                sr = 16000

            audio_float = audio.astype(np.float32)

            if sr != self.target_sr:
                self.logger.debug(f"Resampling audio from {sr}Hz to {self.target_sr}Hz")
                audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=self.target_sr)

            return audio_float, self.target_sr

        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.logger.info(f"Loading audio file: {audio_path}")

        try:
            y, sr = librosa.load(str(audio_path), sr=self.target_sr)
            audio_float = y.astype(np.float32)
            self.logger.debug(f"Audio loaded successfully. Duration: {len(audio_float)/self.target_sr:.2f}s")
            return audio_float, self.target_sr
        except Exception as load_error:
            self.logger.error(f"Failed to load audio file: {str(load_error)}")
            raise

    def apply_normalization(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio / max_val
            self.logger.debug(f"Audio normalized. Max amplitude before: {max_val:.4f}, after: 1.0")
            return audio_normalized
        return audio

    def extract_voice_fingerprint(self, audio: np.ndarray, sr: int, duration_seconds: float = 6.0) -> np.ndarray:
        fingerprint_samples = int(duration_seconds * sr)
        audio_length = len(audio)

        if audio_length < fingerprint_samples:
            self.logger.warning(f"Audio shorter than {duration_seconds}s, padding to match fingerprint duration")
            padded = np.zeros(fingerprint_samples, dtype=audio.dtype)
            padded[:audio_length] = audio
            return padded

        mid_point = audio_length // 2
        start_idx = max(0, mid_point - fingerprint_samples // 2)
        end_idx = min(audio_length, start_idx + fingerprint_samples)

        if end_idx - start_idx < fingerprint_samples:
            start_idx = max(0, audio_length - fingerprint_samples)

        voice_fingerprint = audio[start_idx:end_idx]
        self.logger.debug(f"Voice fingerprint extracted: {len(voice_fingerprint)/sr:.2f}s at position {start_idx/sr:.2f}s")
        return voice_fingerprint

    def detect_vad(self, audio: np.ndarray, sr: int, frame_length: int = 2048,
                   energy_threshold: float = 0.02) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        try:
            if len(audio) < frame_length:
                self.logger.warning("Audio too short for VAD analysis, returning full audio")
                return audio, [(0, len(audio) / sr)]

            hop_length = frame_length // 4
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length)
            energy = np.sqrt(np.mean(S ** 2, axis=0))
            energy_normalized = energy / (np.max(energy) + 1e-8)

            active_frames = energy_normalized > energy_threshold
            frames_time = librosa.frames_to_time(np.arange(len(energy_normalized)), sr=sr, hop_length=hop_length)

            segments = []
            in_segment = False
            start_frame = 0

            for i, is_active in enumerate(active_frames):
                if is_active and not in_segment:
                    start_frame = i
                    in_segment = True
                elif not is_active and in_segment:
                    start_time = float(frames_time[start_frame])
                    end_time = float(frames_time[i])
                    if end_time > start_time:
                        segments.append((start_time, end_time))
                    in_segment = False

            if in_segment:
                segments.append((float(frames_time[start_frame]), float(frames_time[-1])))

            if not segments:
                self.logger.warning("No voice activity detected, using full audio")
                segments = [(0, len(audio) / sr)]

            self.logger.debug(f"Voice activity detection completed. Found {len(segments)} segments")
            return audio, segments

        except Exception as vad_error:
            self.logger.warning(f"Voice activity detection failed: {str(vad_error)}, returning full audio")
            return audio, [(0, len(audio) / sr)]


class FasterWhisperEngine:
    MODEL_MEMORY_REQUIREMENTS = {
        ModelSize.TINY: 390,
        ModelSize.BASE: 880,
        ModelSize.SMALL: 2000,
        ModelSize.MEDIUM: 5000,
        ModelSize.LARGE: 10000,
        ModelSize.LARGEV3: 12000,
    }

    SUPPORTED_LANGUAGES = {
        "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
        "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
        "ru": "Russian", "de": "German",
    }

    VALID_DEVICES = {"cpu", "cuda", "auto"}
    VALID_COMPUTE_TYPES = {ComputeType.INT8, ComputeType.FLOAT16, ComputeType.FLOAT32}

    def __init__(
        self,
        model_size: ModelSize = ModelSize.BASE,
        device: Optional[str] = "auto",
        compute_type: ComputeType = ComputeType.FLOAT32,
        logfile: Optional[str] = None,
        num_workers: int = 1,
        cpu_threads: int = 0,
    ):
        self.logger = LoggerConfig.get_logger(f"FasterWhisperEngine-{model_size.value}", logfile=logfile)
        self.validate_initialization_params(model_size, device, compute_type)

        self.model_size = model_size
        self.compute_type = compute_type
        self.num_workers = num_workers
        self.cpu_threads = cpu_threads
        self.device = self._detect_device(device)
        self.model = None

        self.logger.info(f"Initializing Faster-Whisper engine with model: {model_size.value}")
        self.logger.info(f"Device: {self.device}, Compute Type: {compute_type.value}")

        self.audio_validator = AudioValidator(self.logger)
        self.audio_processor = AudioProcessor(self.logger)

        try:
            self.initialize_model()
            self.logger.info(f"Model {model_size.value} loaded successfully on {self.device}")
        except Exception as init_error:
            self.logger.error(f"Failed to initialize model: {str(init_error)}")
            raise

    def validate_initialization_params(self, model_size: ModelSize, device: str, compute_type: ComputeType) -> None:
        if not isinstance(model_size, ModelSize):
            raise TypeError(f"model_size must be ModelSize enum, got {type(model_size)}")

        if device and device not in self.VALID_DEVICES:
            raise ValueError(f"Invalid device: {device}. Supported: {self.VALID_DEVICES}")

        if compute_type not in self.VALID_COMPUTE_TYPES:
            raise ValueError(f"Invalid compute_type: {compute_type}. Supported: {self.VALID_COMPUTE_TYPES}")

    def _detect_device(self, device: Optional[str]) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                self.logger.info("CUDA device detected and available")
                device_name = torch.cuda.get_device_name(0)
                properties = torch.cuda.get_device_properties(0)
                memory_gb = properties.total_memory / 1e9
                self.logger.info(f"GPU: {device_name}, Memory: {memory_gb:.2f} GB")
                return "cuda"
            else:
                self.logger.warning("CUDA not available, using CPU")
                return "cpu"

        elif device == "cuda":
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            device_name = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            memory_gb = properties.total_memory / 1e9
            self.logger.info(f"Using GPU: {device_name}, Memory: {memory_gb:.2f} GB")
            return "cuda"

        else:
            self.logger.info("Using CPU device")
            return "cpu"

    def initialize_model(self) -> None:
        compute_type_str = self.compute_type.value if isinstance(self.compute_type, ComputeType) else str(self.compute_type)
        self.logger.debug(f"Loading model {self.model_size.value} with compute type {compute_type_str}")

        try:
            self.model = WhisperModel(
                self.model_size.value,
                device=self.device,
                compute_type=compute_type_str,
                num_workers=self.num_workers,
                cpu_threads=self.cpu_threads,
                download_root=None,
                local_files_only=False,
            )
        except ValueError as e:
            if "float16" in str(e) and self.device == "cpu":
                self.logger.warning("float16 not supported on CPU, falling back to float32")
                self.compute_type = ComputeType.FLOAT32
                self.model = WhisperModel(
                    self.model_size.value,
                    device=self.device,
                    compute_type="float32",
                    num_workers=self.num_workers,
                    cpu_threads=self.cpu_threads,
                )
            else:
                raise

        self.logger.debug("Model initialization completed")

    def process_segments(
        self, segments_list: List, sample_rate: int, word_level: bool = False
    ) -> List[TranscriptionSegment]:
        processed_segments = []

        for segment_id, segment in enumerate(segments_list):
            words_list = []

            if word_level and hasattr(segment, 'words') and segment.words:
                for word_info in segment.words:
                    try:
                        word_probability = getattr(word_info, 'probability', 0.0)
                        word_confidence = max(0.0, min(1.0, word_probability))
                        word_obj = TranscriptionWord(
                            word=getattr(word_info, 'word', ''),
                            confidence=word_confidence,
                            start=float(getattr(word_info, 'start', 0.0)),
                            end=float(getattr(word_info, 'end', 0.0)),
                        )
                        words_list.append(word_obj)
                    except (ValueError, TypeError, AttributeError) as word_error:
                        self.logger.warning(f"Failed to process word: {str(word_error)}")
                        continue

            try:
                transcription_segment = TranscriptionSegment(
                    id=segment_id,
                    seek=int(getattr(segment, 'seek', 0)),
                    start=float(getattr(segment, 'start', 0.0)),
                    end=float(getattr(segment, 'end', 0.0)),
                    text=str(getattr(segment, 'text', '')).strip(),
                    tokens=getattr(segment, 'tokens', []),
                    temperature=float(getattr(segment, 'temperature', 0.0)),
                    avg_logprob=float(getattr(segment, 'avg_logprob', 0.0)),
                    compression_ratio=float(getattr(segment, 'compression_ratio', 1.0)),
                    no_speech_prob=float(getattr(segment, 'no_speech_prob', 0.0)),
                    words=words_list,
                )
                processed_segments.append(transcription_segment)
            except (ValueError, TypeError, AttributeError) as seg_error:
                self.logger.warning(f"Failed to process segment {segment_id}: {str(seg_error)}")
                continue

        return processed_segments

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        sr: Optional[int] = None,
        word_level: bool = False,
        vad_filter: bool = True,
        beam_size: int = 5,
        best_of: int = 5,
    ) -> TranscriptionResult:
        start_time = time.time()

        try:
            if isinstance(audio, (str, Path)):
                is_valid, validation_msg = self.audio_validator.validate_file(audio)
                if not is_valid:
                    raise ValueError(validation_msg)
                audio_file_path = str(audio)
            else:
                is_valid, validation_msg = self.audio_validator.validate_array(audio, sr or 16000)
                if not is_valid:
                    raise ValueError(validation_msg)
                audio_file_path = None

            self.logger.info(f"Loading audio: {audio_file_path or 'numpy array'}")

            audio_data, sample_rate = self.audio_processor.load_audio(audio, sr=sr)
            audio_data = self.audio_processor.apply_normalization(audio_data)
            audio_duration = len(audio_data) / sample_rate

            self.logger.info(f"Audio loaded. Duration: {audio_duration:.2f}s, Sample Rate: {sample_rate}Hz")

            voice_fingerprint_sample = self.audio_processor.extract_voice_fingerprint(audio_data, sample_rate)

            if voice_fingerprint_sample is not None:
                expected_len = int(6 * sample_rate)
                actual_len = len(voice_fingerprint_sample)
                if actual_len < expected_len:
                    padded = np.zeros(expected_len, dtype=audio_data.dtype)
                    padded[:actual_len] = voice_fingerprint_sample
                    voice_fingerprint_sample = padded
                    self.logger.warning(f"Fingerprint padded from {actual_len} to {expected_len} samples")

            if vad_filter:
                self.logger.info("Applying voice activity detection")
                audio_data, vad_segments = self.audio_processor.detect_vad(audio_data, sample_rate)
                self.logger.info(f"Detected {len(vad_segments)} speech segments")

            self.logger.info("Starting transcription")

            segments_raw, detected_language = self.model.transcribe(
                audio_data,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                word_timestamps=word_level,
            )

            segments_list = list(segments_raw)
            self.logger.info(f"Transcription completed. Detected language: {detected_language}")
            self.logger.info(f"Number of segments: {len(segments_list)}")

            processed_segments = self.process_segments(segments_list, sample_rate, word_level=word_level)

            valid_segments = [seg for seg in processed_segments if seg.text and seg.text.strip()]

            if not valid_segments:
                full_text = ""
                self.logger.warning("No speech detected in audio")
            else:
                full_text = " ".join(seg.text for seg in valid_segments).strip()

            processing_time = time.time() - start_time

            result = TranscriptionResult(
                text=full_text,
                language=detected_language or language or "unknown",
                duration=audio_duration,
                segments=valid_segments,
                processing_time=processing_time,
                model_size=self.model_size.value,
                device=self.device,
                audio_file=audio_file_path,
                timestamp=datetime.now().isoformat(),
                voice_fingerprint_sample=voice_fingerprint_sample,
            )

            real_time_factor = processing_time / result.duration if result.duration > 0 else 0
            self.logger.info(f"Total processing time: {processing_time:.2f}s")
            self.logger.info(f"Real-time factor: {real_time_factor:.2f}x")

            return result

        except Exception as transcription_error:
            self.logger.error(f"Transcription failed: {str(transcription_error)}", exc_info=True)
            raise

    def batch_transcribe(
        self,
        audio_files: List[Union[str, Path]],
        language: Optional[str] = None,
        word_level: bool = False,
        vad_filter: bool = True,
    ) -> List[Union[TranscriptionResult, Dict]]:
        self.logger.info(f"Starting batch transcription for {len(audio_files)} files")

        max_workers = 2 if self.device == "cuda" else 4
        self.logger.info(f"Using {max_workers} workers for parallel processing")

        results = []
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.transcribe,
                    audio_file,
                    language=language,
                    word_level=word_level,
                    vad_filter=vad_filter,
                ): audio_file
                for audio_file in audio_files
            }

            for future in as_completed(futures):
                audio_file = futures[future]
                try:
                    result = future.result(timeout=600)
                    results.append(result)
                    successful += 1
                    self.logger.info(f"Successfully processed: {audio_file}")
                except Exception as file_error:
                    failed += 1
                    self.logger.warning(f"Failed to process {audio_file}: {str(file_error)}")
                    results.append({
                        "file": str(audio_file),
                        "status": "failed",
                        "error": str(file_error),
                        "timestamp": datetime.now().isoformat(),
                    })

        self.logger.info(f"Batch transcription completed. Successful: {successful}, Failed: {failed}")
        return results

    def get_model_info(self) -> Dict:
        return {
            "model_size": self.model_size.value,
            "device": self.device,
            "compute_type": self.compute_type.value,
            "memory_requirement_mb": self.MODEL_MEMORY_REQUIREMENTS.get(self.model_size, "unknown"),
        }

    def get_supported_languages(self) -> Dict[str, str]:
        return self.SUPPORTED_LANGUAGES.copy()

    def __enter__(self):
        self.logger.debug("Entering context manager")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Cleaning up resources")
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.logger.debug("Model deleted")

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("CUDA cache cleared")
        except Exception as cleanup_error:
            self.logger.warning(f"Error during cleanup: {str(cleanup_error)}")

        return False

    def __del__(self):
        try:
            if hasattr(self, "logger"):
                self.logger.debug("Destructor called")

            if hasattr(self, "model") and self.model is not None:
                del self.model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            if hasattr(self, "logger"):
                self.logger.warning(f"Error during destructor cleanup: {str(cleanup_error)}")


class STTPipeline:
    def __init__(
        self,
        model_size: ModelSize = ModelSize.BASE,
        device: Optional[str] = "auto",
        compute_type: ComputeType = ComputeType.FLOAT32,
        logfile: Optional[str] = None,
    ):
        self.logger = LoggerConfig.get_logger("STTPipeline", logfile=logfile)
        self.engine = FasterWhisperEngine(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            logfile=logfile,
        )
        self.logger.info("STT Pipeline initialized successfully")

    def process_audio(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        sr: Optional[int] = None,
        word_level: bool = False,
        vad_filter: bool = True,
    ) -> Dict:
        try:
            result = self.engine.transcribe(
                audio=audio,
                language=language,
                sr=sr,
                word_level=word_level,
                vad_filter=vad_filter,
            )
            return result.to_dict()
        except Exception as error:
            self.logger.error(f"Audio processing failed: {str(error)}", exc_info=True)
            raise

    def cleanup(self) -> None:
        self.logger.info("Cleaning up STT Pipeline")
        if hasattr(self, "engine"):
            del self.engine