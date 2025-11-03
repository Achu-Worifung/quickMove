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
from dotenv import load_dotenv  # Add this import
from urllib.parse import quote_plus
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

settings = QSettings("MyApp", "AutomataSimulator")
def get_energy_threshold(audio_data, threshold_value=0.05):
    # print('here is the energy threshold', threshold_value)
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
    energy = np.sum(audio_data ** 2) / len(audio_data)  # Calculate energy
    # print('here is the energy', energy)
    return energy > threshold_value

def run_transcription(recording_page, search_Page=None, lineEdit=None, worker_thread=None):
    print('here is the search page', search_Page)
    print('here is the search page', search_Page)
    
    # --------------------loading all the variables from settings------------------
    print('hers is the the line edit', type(lineEdit))
    
    beam_size = int(settings.value('beam'))
    best_of = int(settings.value('best'))
    temperature = float(settings.value('temperature'))
    language = settings.value('language')
    # vad_filter = settings.value('vad_filter')
    energy_threshold = float(settings.value('energy_threshold'))
    # vad_parameters = settings.value('vad_parameters')
    model_size = settings.value('model').lower()
    cpu_cores = int(settings.value('cpu_cores'))
    processing = settings.value('processing')
    CHANNELS = int(settings.value('channel'))
    RATE = int(settings.value('rate'))  # 16kHz sample rate for Whisper
    CHUNK = int(settings.value('chunks'))
    silence_length = float( settings.value('silence'))  # Minimum silence duration for VAD in seconds
    min_record_len = int(settings.value('minlen'))  # Minimum recording length in seconds
    
    max_record_len = int(settings.value('maxlen'))  # Maximum recording length in seconds
    computation_type = "float16" if processing == "GPU" else "int8"
    auto_search_size = int(settings.value('auto_search_size') or 1)
    
    line_edit = recording_page.lineEdit
    
    # -----------------------------loging to debug--------------------------
    
    # print('here is the line edit', line_edit)
    # print('here is the passed page', recording_page)
    
    # Setting up whisper model
    print(f"processing {processing} model {model_size} computational type {computation_type} cpu cores {cpu_cores}" )
    model_size = model_size.lower()
    bible_classifier_model = None
    # Prefer a local cached model directory if it exists; adjust path if your models are stored elsewhere
    # local_model_dir = os.path.normpath(
        # os.path.join(os.path.dirname(__file__), '..', 'models', 'facebook', 'bart-large-mnli')
    # )
    # print('local model dir', local_model_dir)
    # print('is this a valid dir', os.path.isdir(local_model_dir))
    # model_source = local_model_dir if os.path.isdir(local_model_dir) else 'facebook/bart-large-mnli'
    
    model_source = 'finetuned_distilbert_bert'
    auto_search_thread = None
    try:
        bible_classifier_model = pipeline(
            'text-classification',
            model=model_source,
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"Bible classifier model loaded from {model_source}")
    except Exception as e:
        print(f"Failed to load classifier from {model_source}: {e}")
        # # fallback to hub id explicitly
        # try:
        #     bible_classifier_model = pipeline(
        #         'zero-shot-classification',
        #         model='facebook/bart-large-mnli',
        #         device=0 if torch.cuda.is_available() else -1
        #     )
        #     print("Bible classifier model loaded from Hugging Face hub (facebook/bart-large-mnli)")
        pass
        # except Exception as e2:
        #     print(f"Final fallback failed: {e2}")
        #     bible_classifier_model = None
    
    if processing == "GPU" and torch.cuda.is_available():  # Fixed: removed 'not'
        model = WhisperModel(
            model_size,
            device='cuda',
            compute_type=computation_type,
            download_root=f'./models/{model_size}' 
        )
    elif processing == "CPU":
        model = WhisperModel(
            model_size,
            device='cpu',  # Fixed: changed from processing to 'cpu'
            compute_type=computation_type,
            num_workers=cpu_cores,
            cpu_threads=8,
            download_root=f'./models/{model_size}' #where downloaded models are stored
        )
    else:
        # Fallback to CPU if GPU requested but not available
        print("GPU requested but not available, falling back to CPU")
        model = WhisperModel(
            model_size,
            device='cpu',
            compute_type='int8',  # Use int8 for CPU fallback
            num_workers=cpu_cores,
            cpu_threads=8,
            download_root=f'./models/{model_size}'
        )

    # Recording settings
    FORMAT = pyaudio.paInt16

    # Initialize PyAudio and stream variables
    audio = None
    stream = None
    
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
        
        if audio is not None:
            try:
                audio.terminate()
            except:
                pass
        
        # Initialize new audio instance
        audio = pyaudio.PyAudio()
        
        # Try to open the stream with retries
        max_retries = 20
        retry_count = 0
        
        while retry_count < max_retries:
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
                    time.sleep(4)  # Wait 1 second before retrying
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

    try:
        while True:
            frames = []
            silent_frames = 0
            has_speech = False
            
            print("Waiting for speech...")
            
            # Wait for speech to start
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    sound_detected = get_energy_threshold(data, threshold_value=energy_threshold)
                    
                    if sound_detected:
                        # print("Speech detected, recording...")
                        frames.append(data)
                        has_speech = True
                        break
                except Exception as e:
                    print("Error reading audio stream:", e)
                    print("Attempting to reinitialize audio...")
                    
                    # Try to reinitialize the audio stream
                    if initialize_audio():
                        print("Audio reinitialized successfully, continuing...")
                        continue
                    else:
                        print("Failed to reinitialize audio. Waiting before retry...")
                        import time
                        time.sleep(2)
                        if not initialize_audio():
                            print("Audio initialization failed multiple times. Exiting.")
                            return
                        continue
            
            # Continue recording until silence is detected
            while True:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    sound_detected = get_energy_threshold(data, threshold_value=energy_threshold)
                except Exception as e:
                    print("Error reading audio stream during recording:", e)
                    print("Attempting to reinitialize audio...")
                    
                    # Try to reinitialize the audio stream
                    if initialize_audio():
                        print("Audio reinitialized successfully")
                        # Break out of this recording loop to start fresh
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
                
                # If silence detected for a long time and recording length is minimum
                silence_frames_threshold = int(silence_length * RATE / CHUNK)
                if silent_frames >= silence_frames_threshold:
                    if recording_length >= min_record_len:
                        # print(f"Silence detected after {recording_length:.2f} seconds of recording")
                        break
                
                # If we've reached the max recording length break anyway
                if recording_length >= max_record_len:
                    # print(f"Max recording length reached ({max_record_len} seconds)")
                    break
            
            # Save and transcribe the audio only if speech was detected
            if has_speech and len(frames) > 0:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_filename = temp_file.name
                print('here is the temp file', temp_filename)
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
                    
                    # print(f"Transcription for segment {counter}:")
                    segment_text = ""
                    for segment in segments:
                       lineEdit.setText(lineEdit.text() + segment.text.strip() + " ")
                       
                       #classify each segment using the bible classifier
                       result = bible_classifier_model(segment.text.strip())
                       print(f"Segment: {segment.text.strip()}")
                       label = result[0]['label']
                       score = result[0]['score']
                       print(f"Classified as: {label} with score {score}")
                       #search if the label is bible add score baseline after fine tunning
                       if label == 'bible' and worker_thread:
                           perform_auto_search(
                                query=segment.text.strip().lower(), 
                                worker_thread=worker_thread,
                                score=score,
                                max_len=auto_search_size
                            )
                            
                            # auto_search_thrad = SearchThread(
                            #     api_key=os.getenv('API_KEY'),
                            #     engine_id=os.getenv('SEARCH_ENGINE_ID'),
                            #     query=segment.text.strip()
                            # )
                            # auto_search_thrad.finished.connect(perform_auto_search)
                            # auto_search_thrad.start()
                            # print("Started new search thread")
                            # print("auto search thrad", auto_search_thrad)
                            # print("auto search thrad is running", auto_search_thrad.isRunning())
                                
                          
                    if segment_text.strip():
                        line_edit.setText(line_edit.text() + segment_text.strip() + " ")
                    else:
                        # print("No speech detected in this segment.")
                        pass
                        
                    counter += 1
                    
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    pass
                finally:
                    # Clean up the temporary file
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
            
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        # Clean up resources
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except:
                print("Error closing stream")
        if audio is not None:
            try:
                audio.terminate()
            except:
                print("Error terminating audio")

def perform_auto_search(query, worker_thread, score=None, max_len=1):
    if not query or not worker_thread:
        print("Invalid query or worker thread")
        return
    query_full = query.strip() + ' KJV Bible verse' # Use a new var for the full query
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
        print('here are all the items from auto search', items)
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
    results = pyqtSignal(list, str)

    def __init__(self, api_key: str, engine_id: str, query: str):
        super().__init__()
        self.api_key = api_key
        self.engine_id = engine_id
        self.query = query

    def run(self):
        # print("quote_plus(self.query)", quote_plus(self.query))
        url = 'https://customsearch.googleapis.com/customsearch/v1?'
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": quote_plus(self.query  ),  
            "safe": "active",  # Optional
            'hl': 'en',
            'num': 10,
            
            
        }
        response = httpx.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        # print('results for real', results)
        reversed_items = results.get("items", [])[::-1]
        # print('results reversed hello', reversed_items)
        self.finished.emit(reversed_items, self.query)
        # # # print('here are the items',results.get("items", []))