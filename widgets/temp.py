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

# Start preloading in a background thread
threading.Thread(target=load_models_if_needed, daemon=True).start()

def run_transcription(recording_page, search_Page=None, lineEdit=None, worker_thread=None):
    # --- Load Settings ---
    print("Processor: Loading settings...")
    beam_size = int(settings.value('beam') or 5)
    temperature = float(settings.value('temperature') or 0.00)
    language = settings.value('language') or 'en'
    energy_threshold = float(settings.value('energy') or 0.10)
    model_size = (settings.value('model') or 'tiny').lower()
    cpu_cores = int(settings.value('cores') or 1)
    processing = settings.value('processing') or ('CPU' if not torch.cuda.is_available() else 'GPU')
    confidence_threshold = float(percent_to_log_prob(settings.value('confidence_threshold')) or -0.4)
    best_of = int(settings.value('best') or 2)
    auto_search_size = int(settings.value('auto_length') or 1)
    use_prev_context = int(settings.value('prev_context') or 0)


    
    RATE = int(settings.value('rate') or 16000)
    CHUNK = int(settings.value('chunks') or 1024)
    CHANNELS = int(settings.value('channel') or 1)
    FORMAT = pyaudio.paInt16
    
    silence_length = float(settings.value('silence') or 0.5)
    min_record_len = float(settings.value('minlen') or .25)
    max_record_len = float(settings.value('maxlen') or 10) # Increased default max len for usability
    
    # --- Model Setup ---
    whisper_model = None
    classifier = None
    model_source = 'finetuned_distilbert_bert'
    transformers = get_transformers()
    pipeline = get_pipeline()
    torch = get_torch()
    WhisperModel = get_WhisperModel()
    
    worker_thread.loadingStatus.emit()  # Signal to update loading status in GUI
    
    
    #retry logic for lazy import failures
  
    try:
        classifier = pipeline(
            'text-classification',
            model=resource_path(model_source),
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        print(f"Processor: Failed to load classifier: {e}")

    device_type = 'cuda' if processing == "GPU" and torch.cuda.is_available() else 'cpu'
    computation_type = "float16" if device_type == 'cuda' else "int8"
    
    print(f"Processor: Loading Whisper on {device_type}...")
    whisper_model = WhisperModel(
        model_size,
        device=device_type,
        compute_type=computation_type,
        num_workers=cpu_cores,
        download_root=resource_path(os.path.join(f'./models/{model_size}'))
    )

    # --- Audio Initialization Helper ---
    audio = None
    stream = None
    
    def initialize_audio():
        nonlocal audio, stream
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except: pass
        if audio:
            try: audio.terminate()
            except: pass
            
        audio = pyaudio.PyAudio()
        max_retries = 5
        
        for i in range(max_retries):
            try:
                stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                                  input=True, frames_per_buffer=CHUNK)
                print("Audio stream initialized.")
                return True
            except Exception as e:
                print(f"Retry audio init ({i+1}/{max_retries}): {e}")
                time.sleep(1)
        return False

    if not initialize_audio():
        print("Fatal: Could not init audio.")
        return

    # --- Main Loop Variables ---
    print("Recording started...")
    counter = 1
    
    # Store the last few words from the previous segment
    previous_context = "" 
    CONTEXT_WORD_LIMIT = 4 if use_prev_context == 1 else 0

    try:
        while True:
            frames = []
            has_speech = False
            
            print("Waiting for speech...")
            
            # 1. Listen for initial sound
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    if get_energy_threshold(data, threshold_value=energy_threshold):
                        frames.append(data)
                        has_speech = True
                        break
                except Exception as e:
                    print(f"Stream error (wait): {e}")
                    initialize_audio()
            
            # 2. Record until silence or max length
            silent_frames = 0
            silence_frames_threshold = int(silence_length * RATE / CHUNK)
            
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    if get_energy_threshold(data, threshold_value=energy_threshold):
                        silent_frames = 0
                    else:
                        silent_frames += 1
                    
                    recording_seconds = len(frames) * CHUNK / RATE
                    
                    if silent_frames >= silence_frames_threshold and recording_seconds >= min_record_len:
                        break
                    if recording_seconds >= max_record_len:
                        break
                        
                except Exception as e:
                    print(f"Stream error (record): {e}")
                    initialize_audio()
                    break

            # 3. Transcribe and Classify
            if has_speech and len(frames) > 0:
                temp_filename = ""
                try:
                    # Create temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_filename = temp_file.name
                    
                    with wave.open(temp_filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                    
                    # Transcribe
                    segments, _ = whisper_model.transcribe(
                        temp_filename,
                        beam_size=beam_size,
                        temperature=temperature,
                        language=language,
                        vad_filter=True,
                        best_of=best_of,
                        initial_prompt = previous_context if use_prev_context == 1 else "",
                        vad_parameters=dict(min_silence_duration_ms=500),
                        log_prob_threshold=confidence_threshold
                    )
                    
                    full_trans = " ".join([s.text for s in segments]).strip()
                    worker_thread.guitextReady.emit(full_trans)  # Emit the transcribed text to update the GUI

                    if full_trans:
                        # --- CONTEXT LOGIC START ---
                        # Combine previous context with current text for classification
                        text_to_classify = f"{previous_context} {full_trans}".strip()
                        
                        print(f"Transcribed: '{full_trans}'")
                        print(f"Classifying Context: '{text_to_classify}'") # Debugging context

                        if classifier:
                            result = classifier(text_to_classify)
                            label = result[0]['label']
                            score = result[0]['score']
                            
                            print(f"Result: {label} ({score:.2f})")
                            
                            if label == 'bible' and worker_thread:
                                # We usually search the full transcription, 
                                # but you can pass text_to_classify if you want the context included in the Google search too.
                                perform_auto_search(text_to_classify, score, auto_search_size, worker_thread)

                        # Update context for the NEXT loop (Last 4 words)
                        words = full_trans.split()
                        if len(words) > CONTEXT_WORD_LIMIT:
                            previous_context = " ".join(words[-CONTEXT_WORD_LIMIT:])
                        else:
                            previous_context = full_trans

                    counter += 1

                except Exception as e:
                    print(f"Processing error: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        try: os.remove(temp_filename)
                        except: pass

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        if stream: stream.close()
        if audio: audio.terminate()

def perform_auto_search(query, confidence, max_len, worker_thread):
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