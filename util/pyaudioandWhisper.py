import pyaudio
import numpy as np
import wave
import os
from PyQt5.QtCore import QSettings
import tempfile
import httpx
from urllib.parse import quote_plus
from util.util import resource_path
import math
import time
import threading
import importlib
import webrtcvad



settings = QSettings("MyApp", "AutomataSimulator")
_torch = None
_transformers = None
_pipeline = None
_WhisperModel = None

# Cache for modules imported by lazy_import to avoid repeated imports and retries
_lazy_import_cache = {}
def get_energy_threshold(audio_data, threshold_value=0.05):
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.sum(audio_data ** 2) / len(audio_data)
    return energy > threshold_value

def percent_to_log_prob(percent):
    if percent is None: return -0.40
    if percent <= 0: return float('-inf')
    if percent >= 100: return 0.0
    return math.log(percent / 100.0)
def lazy_import(module_name, max_retries=3, delay=2):
    """
    Lazily import a module with retry logic.
    
    :param module_name: The name of the module to import.
    :param max_retries: Maximum number of retries for the import.
    :param delay: Delay (in seconds) between retries.
    :return: The imported module.
    """
    if module_name in _lazy_import_cache:
        return _lazy_import_cache[module_name]

    last_exception = None
    for attempt in range(max_retries):
        try:
            module = importlib.import_module(module_name)
            _lazy_import_cache[module_name] = module
            return module
        except Exception as e:
            last_exception = e
            print(f"Retry {attempt + 1}/{max_retries} for module '{module_name}' failed: {e}")
            time.sleep(delay)

    raise ImportError(f"Failed to import module '{module_name}' after {max_retries} retries") from last_exception

# Specific lazy import functions
def get_torch():
    return lazy_import("torch")

def get_transformers():
    transformers = lazy_import("transformers")
    # Suppress irrelevant warnings from transformers
    transformers.utils.logging.set_verbosity_error()
    return transformers

def get_WhisperModel():
    faster_whisper = lazy_import("faster_whisper")
    return faster_whisper.WhisperModel

def get_pipeline():
    transformers = get_transformers()
    return transformers.pipeline

# Preload models in a background thread
def load_models_if_needed():
    try:
        get_torch()
        get_transformers()
        get_WhisperModel()
        get_pipeline()
    except Exception as e:
        print(f"Error during model preloading: {e}")
def get_classifier(source, pipeline=get_pipeline()):
    
    try:
        classifier = pipeline (
                'text-classification',
                model=resource_path(source),
                device=0 if get_torch().cuda.is_available() else -1
        )
    except Exception as e:
        print(f"Processor: Failed to load classifier: {e}")
        classifier = None
    return classifier


# Start preloading in a background thread
threading.Thread(target=load_models_if_needed, daemon=True).start()
import re

def extract_stable_sentences(text):
    """Return only completed sentences (drop trailing fragment)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        return ""
    return " ".join(sentences[:-1])

def frames_overlap(frames, rate, chunk, overlap_sec=0.7):
    overlap_frames = int(overlap_sec * rate / chunk)
    if len(frames) > overlap_frames:
        return frames[-overlap_frames:]
    return frames

def run_transcription(recording_page, search_Page=None, lineEdit=None, worker_thread=None):
    print("Processor: Loading settings...")

    beam_size = 1                       # REAL-TIME SAFE
    best_of = 1
    temperature = 0.0

    language = settings.value('language') or 'en'
    energy_threshold = float(settings.value('energy') or 0.10)
    model_size = (settings.value('model') or 'tiny').lower()
    cpu_cores = int(settings.value('cores') or 1)
    confidence_threshold = float(percent_to_log_prob(settings.value('confidence_threshold')) or -0.4)
    auto_search_size = int(settings.value('auto_length') or 1)

    RATE = int(settings.value('rate') or 16000)
    CHUNK = int(settings.value('chunks') or 1024)
    CHANNELS = int(settings.value('channel') or 1)
    FORMAT = pyaudio.paInt16

    silence_length = float(settings.value('silence') or 0.6)
    min_record_len = float(settings.value('minlen') or 0.25)
    max_record_len = float(settings.value('maxlen') or 20)  # allow full thoughts
    
    classifier_src = 'finetuned_distilbert'
    torch = get_torch()
    WhisperModel = get_WhisperModel()
    classifier = get_classifier(classifier_src)
    worker_thread.loadingStatus.emit()  # Signal to update loading status in GUI


    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    computation_type = "float16" if device_type == 'cuda' else "int8"
    
    

    whisper_model = WhisperModel(
        model_size,
        device=device_type,
        compute_type=computation_type,
        num_workers=cpu_cores,
        download_root=resource_path(os.path.join(f'./models/{model_size}'))
    )
    vad = webrtcvad.Vad(3)  # Set aggressiveness mode (0-3)



    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

    previous_audio_overlap = []

    # VAD requires specific frame sizes: 10, 20, or 30ms at 8k, 16k, 32k, or 48k Hz
    vad_frame_duration = 30  # ms (30ms works well for real-time)
    vad_frame_size = int(RATE * vad_frame_duration / 1000) * 2  # 2 bytes per sample (16-bit)
    speech_detected_count = 0
    
    try:
        while True:
            frames = []
            silence_frames = 0
            max_silence_frames = int(silence_length * 1000 / vad_frame_duration)
            
            print(f"Waiting for speech... (frame size: {vad_frame_size} bytes, aggressiveness: 2)")

            # Wait for initial speech detection
            while True:
                audio_chunk = stream.read(int(RATE * vad_frame_duration / 1000), exception_on_overflow=False)
                
                if len(audio_chunk) == vad_frame_size:
                    is_speech = vad.is_speech(audio_chunk, RATE)
                    if is_speech:
                        speech_detected_count += 1
                        if speech_detected_count >= 2:  # Require 2 consecutive speech frames
                            print(f"✓ Speech detected! Recording...")
                            frames.append(audio_chunk)
                            break
                    else:
                        speech_detected_count = 0
                else:
                    print(f"⚠ Chunk size mismatch: got {len(audio_chunk)}, expected {vad_frame_size}")
                    speech_detected_count = 0

            # Record while speech is present (with silence timeout)
            recording_start = time.time()
            while True:
                audio_chunk = stream.read(int(RATE * vad_frame_duration / 1000), exception_on_overflow=False)
                frames.append(audio_chunk)
                
                recording_duration = time.time() - recording_start
                
                # Check for speech in this frame
                if len(audio_chunk) == vad_frame_size:
                    is_speech = vad.is_speech(audio_chunk, RATE)
                    if is_speech:
                        silence_frames = 0  # Reset silence counter
                    else:
                        silence_frames += 1
                else:
                    # If we can't validate with VAD, be conservative
                    silence_frames += 1
                
                # Stop conditions (check min length before allowing silence stop)
                if recording_duration >= max_record_len:
                    print(f"Max recording length reached ({max_record_len}s)")
                    break
                
                if recording_duration >= min_record_len and silence_frames >= max_silence_frames:
                    print(f"Silence detected after {recording_duration:.2f}s ({silence_frames} silent frames)")
                    break

            # Skip if recording is too short
            if recording_duration < min_record_len:
                print(f"Recording too short ({recording_duration:.2f}s), skipping...")
                continue

            print(f"Recording finished: {recording_duration:.2f}s, {len(frames)} frames")

            # 🔁 AUDIO OVERLAP (instead of text context)
            frames = previous_audio_overlap + frames

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

            segments, _ = whisper_model.transcribe(
                temp_filename,
                language=language,
                beam_size=1, #change this later
                best_of=1,
                temperature=0.0,
                vad_filter=False,                   
                log_prob_threshold=confidence_threshold
            )

            raw_text = " ".join(s.text for s in segments).strip()
            stable_text = extract_stable_sentences(raw_text)
            print(f"Transcribed: {stable_text}")
            print(f"Stable text: {stable_text}")
            worker_thread.guitextReady.emit(stable_text)

            if stable_text:
                worker_thread.guitextReady.emit(stable_text)
                
                if classifier:
                    try:
                        classification = classifier(stable_text, truncation=True)
                        label = classification[0]['label']
                        score = classification[0]['score']
                        print(f"Classification: {label} (score: {score:.4f})")
                    except Exception as e:
                        print(f"Error during classification: {e}")
                    

                if worker_thread and auto_search_size > 0 and label == 'bible': #only search if classified as bible
                    perform_auto_search(stable_text, 1.0, auto_search_size, worker_thread)

            # Prepare overlap for next iteration
            previous_audio_overlap = frames_overlap(frames, RATE, CHUNK)

            os.remove(temp_filename)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        stream.close()
        audio.terminate()

def perform_auto_search(query, confidence, max_len, worker_thread):
    print(f"Performing auto-search for query: '{query}', confidence: {confidence}, max_len: {max_len}, worker_thread: {worker_thread is not None}")
    if not query or not worker_thread: return
    
    # We append KJV Bible verse to ensure relevance
    query_full = query.strip() + ' KJV Bible verse' 
    
    url = 'https://customsearch.googleapis.com/customsearch/v1?'
    params = {
        "key": os.environ.get('API_KEY'),
        "cx": os.environ.get('SEARCH_ENGINE_ID'),
        "q": quote_plus(query_full),
        "safe": "active",  
        'hl': 'en',
        'num': 10,
    }
    try:
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        results = response.json()
        items = results.get("items", [])
        print(f"Auto-search found {len(items)} items for query: '{query_full}'")
        
        if items and worker_thread:
             # This signal is defined in TranscriptionWorker in your other file
             worker_thread.autoSearchResults.emit(items, query_full, confidence, max_len)
    except httpx.TimeoutException:
         print("Auto-search timed out.")
    except Exception as e:
        print(f"Error in auto-search: {e}")