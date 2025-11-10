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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import transformers
transformers.utils.logging.set_verbosity_error()
from util.util import resource_path

basedir = os.path.dirname(__file__)
settings = QSettings("MyApp", "AutomataSimulator")

def get_energy_threshold(audio_data, threshold_value=0.05):
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.sum(audio_data ** 2) / len(audio_data)
    return energy > threshold_value

class TranscriptionController:
    """Controller class to manage transcription lifecycle"""
    def __init__(self):
        self.should_stop = False
        self.lock = threading.Lock()
    
    def stop(self):
        with self.lock:
            self.should_stop = True
    
    def is_stopped(self):
        with self.lock:
            return self.should_stop

def run_transcription(recording_page, search_Page=None, lineEdit=None, worker_thread=None, controller=None):
    """
    Main transcription function with proper resource cleanup.
    Pass a TranscriptionController instance to enable graceful shutdown.
    """
    print('here is the search page', search_Page)
    
    if controller is None:
        controller = TranscriptionController()
    
    # --------------------loading all the variables from settings------------------
    beam_size = int(settings.value('beam') or 5)
    best_of = int(settings.value('best') or 2)
    temperature = float(settings.value('temperature') or 0.00)
    language = settings.value('language') or 'en'
    energy_threshold = float(settings.value('energy_threshold') or 0.10)
    model_size = (settings.value('model') or 'tiny').lower()
    cpu_cores = int(settings.value('cpu_cores') or 1)
    processing = settings.value('processing') or ('CPU' if not torch.cuda.is_available() else 'GPU')
    
    # Recording settings
    CHANNELS = int(settings.value('channel') or 1)
    RATE = int(settings.value('rate') or 16000)
    CHUNK = int(settings.value('chunks') or 1024)
    silence_length = float(settings.value('silence') or 0.5)
    min_record_len = int(settings.value('minlen') or 1)
    max_record_len = int(settings.value('maxlen') or 5)
    computation_type = "float16" if processing == "GPU" else "int8"
    auto_search_size = int(settings.value('auto_length') or 1)
    
    line_edit = recording_page.lineEdit
    
    # Setting up whisper model
    print(f"processing {processing} model {model_size} computational type {computation_type} cpu cores {cpu_cores}")
    
    model = None
    bible_classifier_model = None
    audio = None
    stream = None
    
    try:
        # Load classifier model
        model_source = 'finetuned_distilbert_bert'
        try:
            bible_classifier_model = pipeline(
                'text-classification',
                model=resource_path(model_source),
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"Bible classifier model loaded from {model_source}")
        except Exception as e:
            print(f"Failed to load classifier from {model_source}: {e}")
        
        # Load Whisper model
        if processing == "GPU" and torch.cuda.is_available():
            model = WhisperModel(
                model_size,
                device='cuda',
                compute_type=computation_type,
                download_root=resource_path(os.path.join(f'./models/{model_size}'))
            )
        elif processing == "CPU":
            model = WhisperModel(
                model_size,
                device='cpu',
                compute_type=computation_type,
                num_workers=cpu_cores,
                cpu_threads=8,
                download_root=resource_path(os.path.join(f'./models/{model_size}'))
            )
        else:
            print("GPU requested but not available, falling back to CPU")
            model = WhisperModel(
                model_size,
                device='cpu',
                compute_type='int8',
                num_workers=cpu_cores,
                cpu_threads=8,
                download_root=resource_path(os.path.join(f'./models/{model_size}'))
            )

        FORMAT = pyaudio.paInt16
        
        def initialize_audio():
            """Initialize or reinitialize the audio stream"""
            nonlocal audio, stream
            
            # Clean up existing resources
            if stream is not None:
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except:
                    pass
                stream = None
            
            if audio is not None:
                try:
                    audio.terminate()
                except:
                    pass
                audio = None
            
            # Check if we should stop before initializing
            if controller.is_stopped():
                return False
            
            # Initialize new audio instance
            audio = pyaudio.PyAudio()
            
            # Try to open the stream with retries
            max_retries = 20
            retry_count = 0
            
            while retry_count < max_retries and not controller.is_stopped():
                try:
                    stream = audio.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK
                    )
                    print("Audio stream initialized successfully")
                    return True
                except Exception as e:
                    retry_count += 1
                    print(f"Failed to initialize audio stream (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        import time
                        time.sleep(4)
                    else:
                        print("Max retries reached. Could not initialize audio stream.")
                        return False
            
            return False

        # Initial audio setup
        if not initialize_audio():
            print("Failed to initialize audio. Exiting.")
            return

        print("Recording...")
        counter = 1
        prev_segment_text = ""

        # Main recording loop with stop check
        while not controller.is_stopped():
            frames = []
            silent_frames = 0
            has_speech = False
            
            print("Waiting for speech...")
            
            # Wait for speech to start
            while not controller.is_stopped():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    sound_detected = get_energy_threshold(data, threshold_value=energy_threshold)
                    
                    if sound_detected:
                        frames.append(data)
                        has_speech = True
                        break
                except Exception as e:
                    print("Error reading audio stream:", e)
                    print("Attempting to reinitialize audio...")
                    
                    if initialize_audio():
                        print("Audio reinitialized successfully, continuing...")
                        continue
                    else:
                        print("Failed to reinitialize audio. Exiting.")
                        return
            
            # Check if we should stop after waiting for speech
            if controller.is_stopped():
                break
            
            # Continue recording until silence is detected
            while not controller.is_stopped():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    sound_detected = get_energy_threshold(data, threshold_value=energy_threshold)
                except Exception as e:
                    print("Error reading audio stream during recording:", e)
                    print("Attempting to reinitialize audio...")
                    
                    if initialize_audio():
                        print("Audio reinitialized successfully")
                        break
                    else:
                        print("Failed to reinitialize audio during recording")
                        import time
                        time.sleep(2)
                        continue
                
                recording_length = len(frames) * CHUNK / RATE
                frames.append(data)
                
                if sound_detected:
                    silent_frames = 0
                else:
                    silent_frames += 1
                
                silence_frames_threshold = int(silence_length * RATE / CHUNK)
                if silent_frames >= silence_frames_threshold:
                    if recording_length >= min_record_len:
                        break
                
                if recording_length >= max_record_len:
                    break
            
            # Check if we should stop before processing
            if controller.is_stopped():
                break
            
            # Save and transcribe the audio only if speech was detected
            if has_speech and len(frames) > 0:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_filename = temp_file.name
                temp_file.close()
                
                try:
                    with wave.open(temp_filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                    
                    print(f"Transcribing segment {counter}...")
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
                    
                    full_trans = [segment.text.strip().lower() for segment in segments]
                    full_text = " ".join(full_trans)
                    lineEdit.setText(lineEdit.text() + " " + full_text + " ")
                    
                    # Classify if model available
                    if full_trans and bible_classifier_model:
                        result = bible_classifier_model(prev_segment_text + " " + full_text.lower())
                        label = result[0]['label']
                        score = result[0]['score']
                        
                        print(f"Classified as: {label} with score {score}")
                        
                        if label == 'bible' and worker_thread:
                            perform_auto_search(
                                query=prev_segment_text + " " + full_text.lower(),
                                worker_thread=worker_thread,
                                score=score,
                                max_len=auto_search_size
                            )
                    
                    if full_text.strip():
                        line_edit.setText(line_edit.text() + full_text.strip() + " ")
                    
                    prev_segment_text = full_text
                    counter += 1
                    
                except Exception as e:
                    print(f"Error during transcription: {e}")
                finally:
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
        
        print("Transcription stopped gracefully.")
        
    except KeyboardInterrupt:
        print("Recording stopped by keyboard interrupt.")
    except Exception as e:
        print(f"Unexpected error in transcription: {e}")
    finally:
        print("Cleaning up resources...")
        
        # Clean up audio resources FIRST
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                print("Stream closed")
            except Exception as e:
                print(f"Error closing stream: {e}")
        
        if audio is not None:
            try:
                audio.terminate()
                print("PyAudio terminated")
            except Exception as e:
                print(f"Error terminating audio: {e}")
        
        # Clean up model AFTER audio
        if model is not None:
            try:
                del model
                print("Whisper model cleaned up")
            except Exception as e:
                print(f"Error cleaning up model: {e}")
        
        # Clean up classifier
        if bible_classifier_model is not None:
            try:
                del bible_classifier_model
                print("Classifier model cleaned up")
            except Exception as e:
                print(f"Error cleaning up classifier: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
        
        print("Cleanup complete")

def perform_auto_search(query, worker_thread, score=None, max_len=1):
    if not query or not worker_thread:
        print("Invalid query or worker thread")
        return
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
        response = httpx.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        items = results.get("items", [])
        print('here is the length of items', len(items))
        reversed_items = items[::-1]
        if reversed_items and worker_thread:
            print('here is the top result from auto search', items[0])
            worker_thread.autoSearchResults.emit(items[:max_len], query_full, score)
        
    except httpx.HTTPStatusError as e:
        print(f"Auto-search HTTP error: {e}")
    except Exception as e:
        print(f"Error in auto-search: {e}")

class SearchResultWorker(QThread):
    finish = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        load_dotenv(self.basedir + '/.env')
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('SEARCH_ENGINE_ID')
        
    def run(self):
        pass
        
    def performSearch(self):
        self.search_thread = SearchThread(self.api_key, self.engine_id, self.parent.line_edit.text())
        self.search_thread.finished.connect(self.handle_search_results)
        self.search_thread.start()
        
    def handle_search_results(self, results):
        print('here are the results', results)
        pass

class SearchThread(QThread):
    finished = pyqtSignal(list, str)

    def __init__(self, api_key: str, engine_id: str, query: str):
        super().__init__()
        self.api_key = api_key
        self.engine_id = engine_id
        self.query = query

    def run(self):
        url = 'https://customsearch.googleapis.com/customsearch/v1?'
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": self.query + ' KJV',
            "safe": "active",
            'hl': 'en',
            'num': 10,
        }
        try:
            response = httpx.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            reversed_items = results.get("items", [])[::-1]
            self.finished.emit(reversed_items, self.query)
        except Exception as e:
            print(f"Error in search thread: {e}")
            self.finished.emit([], self.query)