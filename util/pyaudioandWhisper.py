import threading
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import torch
import wave
import os
from PyQt5.QtCore import QSettings, QThread, pyqtSignal
import tempfile
import httpx
from dotenv import load_dotenv
from urllib.parse import quote_plus
from transformers import pipeline
import transformers
import gc
import time

# Suppress mostly irrelevant transformer warnings
transformers.utils.logging.set_verbosity_error()

from util.util import resource_path

basedir = os.path.dirname(__file__)
settings = QSettings("MyApp", "AutomataSimulator")

# --- Helper Functions ---

def get_energy_threshold(audio_data, threshold_value=0.05):
    """Calculates if audio chunk energy exceeds threshold."""
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.sum(audio_data ** 2) / len(audio_data)
    return energy > threshold_value

def cleanup_audio_resources(audio_instance, stream_instance):
    """Safely closes audio stream and terminates PyAudio instance."""
    if stream_instance is not None:
        try:
            if stream_instance.is_active():
                stream_instance.stop_stream()
            stream_instance.close()
            print("Audio stream closed.")
        except Exception as e:
            print(f"Error closing stream: {e}")
            
    if audio_instance is not None:
        try:
            audio_instance.terminate()
            print("PyAudio terminated.")
        except Exception as e:
            print(f"Error terminating PyAudio: {e}")

def cleanup_model_resources(model_instance, classifier_instance):
    """Safely deletes models and clears GPU cache."""
    if model_instance is not None:
        try:
            del model_instance
            print("Whisper model deleted.")
        except Exception as e:
            print(f"Error deleting whisper model: {e}")

    if classifier_instance is not None:
        try:
            del classifier_instance
            print("Classifier model deleted.")
        except Exception as e:
            print(f"Error deleting classifier: {e}")

    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

def perform_auto_search(query, worker_thread, score=None, max_len=1):
    """Executes a Google Custom Search and emits results to the main thread."""
    if not query or not worker_thread:
        return

    query_full = query.strip() + ' KJV Bible verse'
    url = 'https://customsearch.googleapis.com/customsearch/v1?'
    params = {
        "key": os.getenv('API_KEY'),
        "cx": os.getenv('SEARCH_ENGINE_ID'),
        "q": quote_plus(query_full),
        "safe": "active",
        'hl': 'en',
        'num': 10,
    }
    
    try:
        # Use a timeout to prevent hanging indefinetly
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        results = response.json()
        
        items = results.get("items", [])
        # print(f'Auto-search found {len(items)} items')
        
        # Reverse as per your original logic (assuming you want bottom-up results?)
        # If you want top relevance, usually you don't reverse. 
        # Keeping it as is to match your original logic.
        if items and worker_thread:
             worker_thread.autoSearchResults.emit(items[:max_len], query_full, score)
        
    except httpx.TimeoutException:
         print("Auto-search timed out.")
    except Exception as e:
        print(f"Error in auto-search: {e}")

# --- Controller Class ---

class TranscriptionController:
    """Controller class to manage transcription lifecycle thread-safely."""
    def __init__(self):
        self._should_stop = False
        self._lock = threading.Lock()
    
    def stop(self):
        with self._lock:
            self._should_stop = True
    
    def is_stopped(self):
        with self._lock:
            return self._should_stop

# --- Main Runner ---

def run_transcription(recording_page, search_Page=None, lineEdit=None, worker_thread=None, controller=None):
    """
    Main transcription loop.
    Monitors audio, detects speech (VAD), records segments, 
    transcribes them, classifies them, and triggers auto-search.
    """
    if controller is None:
        controller = TranscriptionController()

    # Load environment variables for auto-search
    load_dotenv(os.path.join(basedir, '.env'))

    # --- Settings Loading ---
    beam_size = int(settings.value('beam') or 5)
    best_of = int(settings.value('best') or 2)
    temperature = float(settings.value('temperature') or 0.00)
    language = settings.value('language') or 'en'
    energy_threshold = float(settings.value('energy_threshold') or 0.10)
    model_size = (settings.value('model') or 'tiny').lower()
    cpu_cores = int(settings.value('cpu_cores') or 1)
    processing = settings.value('processing') or ('CPU' if not torch.cuda.is_available() else 'GPU')

    CHANNELS = int(settings.value('channel') or 1)
    RATE = int(settings.value('rate') or 16000)
    CHUNK = int(settings.value('chunks') or 1024)
    silence_length = float(settings.value('silence') or 0.5)
    min_record_len = int(settings.value('minlen') or 1)
    max_record_len = int(settings.value('maxlen') or 5)
    computation_type = "float16" if processing == "GPU" else "int8"
    auto_search_size = int(settings.value('auto_length') or 1)

    print(f"Starting transcription: {processing} | Model: {model_size} | Type: {computation_type}")
    #check if the model has been downloaded before 

    # --- Resources ---
    model = None
    bible_classifier_model = None
    audio = None
    stream = None

    try:
        # 1. Load Models
        # -----------------------------
        # Classifier
        model_source = 'finetuned_distilbert_bert'
        try:
            bible_classifier_model = pipeline(
                'text-classification',
                model=resource_path(model_source),
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"Classifier loaded from {model_source}")
        except Exception as e:
            print(f"Failed to load classifier: {e}")

        # Whisper
        device_type = 'cuda' if processing == "GPU" and torch.cuda.is_available() else 'cpu'
        # Fallback if GPU requested but missing
        if processing == "GPU" and device_type == 'cpu':
             print("GPU requested but not available, falling back to CPU int8")
             computation_type = "int8"

        model = WhisperModel(
            model_size,
            device=device_type,
            compute_type=computation_type,
            num_workers=cpu_cores,
            download_root=resource_path(os.path.join(f'./models/{model_size}'))
        )

        # 2. Initialize Audio
        # -----------------------------
        FORMAT = pyaudio.paInt16
        
        def initialize_audio_stream():
            """Inner helper to (re)initialize audio stream with retries."""
            nonlocal audio, stream
            # Clean up any existing first
            cleanup_audio_resources(audio, stream)
            audio, stream = None, None

            if controller.is_stopped(): return False

            audio_inst = pyaudio.PyAudio()
            stream_inst = None
            
            max_retries = 5 # Reduced from 20, 5 is usually enough to know if it's broken
            for i in range(max_retries):
                if controller.is_stopped(): return False
                try:
                    stream_inst = audio_inst.open(
                        format=FORMAT, channels=CHANNELS, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK
                    )
                    print("Audio stream initialized.")
                    # Assign back to nonlocal variables on success
                    audio, stream = audio_inst, stream_inst
                    return True
                except Exception as e:
                    print(f"Audio init failed (attempt {i+1}/{max_retries}): {e}")
                    time.sleep(2)
            
            # If we failed all retries, clean up the audio instance
            if audio_inst: audio_inst.terminate()
            return False

        if not initialize_audio_stream():
             print("Could not initialize audio after retries. Exiting thread.")
             return

        # 3. Main Recording Loop
        # -----------------------------
        print("Ready to record...")
        counter = 1
        prev_segment_text = ""
        silence_frames_threshold = int(silence_length * RATE / CHUNK)

        while not controller.is_stopped():
            frames = []
            silent_frames = 0
            is_recording_speech = False
            
            print("Waiting for speech...")

            # Inner loop: handles BOTH waiting for speech AND recording speech
            while not controller.is_stopped():
                try:
                    # Non-blocking read if possible, though standard PyAudio read is blocking.
                    # exception_on_overflow=False prevents crashing if computer is slow for a moment.
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    is_loud = get_energy_threshold(data, threshold_value=energy_threshold)

                    if not is_recording_speech:
                        # STATE: Waiting for speech
                        if is_loud:
                            print("Speech detected, starting segment recording...")
                            is_recording_speech = True
                            frames.append(data)
                            silent_frames = 0
                        # else: just continue waiting
                    else:
                        # STATE: Recording speech
                        frames.append(data)
                        if is_loud:
                            silent_frames = 0
                        else:
                            silent_frames += 1
                        
                        recording_length_sec = (len(frames) * CHUNK) / RATE

                        # Check stop conditions
                        if silent_frames >= silence_frames_threshold:
                            if recording_length_sec >= min_record_len:
                                # print("Silence detected, finishing segment.")
                                break # Exit inner loop to process
                            # else: keep recording, segment too short

                        if recording_length_sec >= max_record_len:
                            # print("Max length reached, finishing segment.")
                            break # Exit inner loop to process

                except (OSError, IOError) as e:
                    print(f"Audio stream error: {e}")
                    print("Attempting to reinitialize audio...")
                    if not initialize_audio_stream():
                        print("Failed to recover audio. Exiting loop.")
                        return # Exit main function entirely
                    # If recovered, continue the outer loop (resetting frames for this segment)
                    break 

            # END Inner Loop

            if controller.is_stopped():
                break

            # 4. Process Recording (If we have valid frames)
            # -----------------------------
            if is_recording_speech and len(frames) > 0:
                # Use context manager for temp file to ensure it closes
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                try:
                    # Write WAV
                    with wave.open(temp_filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))

                    # Transcribe
                    # print(f"Transcribing segment {counter}...")
                    segments, _ = model.transcribe(
                        temp_filename,
                        beam_size=beam_size,
                        best_of=best_of,
                        temperature=temperature,
                        language=language,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        word_timestamps=False
                    )

                    full_trans = [s.text.strip() for s in segments]
                    full_text = " ".join(full_trans)
                    current_text_lower = full_text.lower()

                    if full_text:
                         # Update UI immediately
                         if lineEdit:
                             # Use simple concatenation for now, maybe consider signals if this causes threading issues later
                             current_ui_text = lineEdit.text()
                             lineEdit.setText(current_ui_text + " " + full_text)

                         # Classify
                         if bible_classifier_model:
                             # Combine with previous segment for better context
                             context_text = (prev_segment_text + " " + current_text_lower).strip()
                             result = bible_classifier_model(context_text)
                             label = result[0]['label']
                             score = result[0]['score']
                             
                             # print(f"Segment {counter} classified as: {label} ({score:.2f})")
                             
                             if label == 'bible' and worker_thread:
                                 perform_auto_search(
                                     query=context_text,
                                     worker_thread=worker_thread,
                                     score=score,
                                     max_len=auto_search_size
                                 )

                         prev_segment_text = current_text_lower
                         counter += 1

                except Exception as e:
                    print(f"Error processing segment {counter}: {e}")
                finally:
                    # Always clean up temp file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)

    except KeyboardInterrupt:
        print("Transcription stopped by keyboard.")
    except Exception as e:
        print(f"Critical error in transcription loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down transcription thread...")
        # Use the consolidated cleanup functions
        cleanup_audio_resources(audio, stream)
        cleanup_model_resources(model, bible_classifier_model)
        print("Transcription thread completely finished.")