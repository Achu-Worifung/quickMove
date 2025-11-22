import threading
import pyaudio
import numpy as np

import wave
import os
from PyQt5.QtCore import QSettings, QThread, pyqtSignal, pyqtSlot
import tempfile
import httpx
from dotenv import load_dotenv
from urllib.parse import quote_plus

import gc
import time
import math
from queue import Queue, Empty



from util.util import resource_path

basedir = os.path.dirname(__file__)
settings = QSettings("MyApp", "AutomataSimulator")

# --- Helper Functions ---

#lazy imports for torch, transformers, WhisperModel, pipeline (improve startup time)
_torch = None
_transformers = None
_WhisperModel = None 
_pipeline = None

import math


def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch
def get_transformers():
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
        # Suppress mostly irrelevant transformer warnings
        transformers.utils.logging.set_verbosity_error()
    return _transformers
def get_WhisperModel():
    global _WhisperModel
    if _WhisperModel is None:
        from faster_whisper import WhisperModel
        _WhisperModel = WhisperModel
    return _WhisperModel
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            _pipeline = pipeline
        except ImportError as e:
            print(f"Error importing 'pipeline' from 'transformers': {e}")
            _pipeline = None
    return _pipeline
def percent_to_log_prob(percent):
    """
    Convert percentage (0-100) to log probability
    """
    if percent is None:
        return -0.40
    if percent <= 0:
        return float('-inf')
    if percent >= 100:
        return 0.0
    
    probability = percent / 100.0
    log_prob = math.log(probability)
    return log_prob

def get_energy_threshold(audio_data, threshold_value=0.05):
    """Calculates if audio chunk energy exceeds threshold."""
    if not audio_data:
        return False
    audio_data_fp32 = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.sum(audio_data_fp32 ** 2) / len(audio_data_fp32)
    # print(f"Calculated energy: {energy}, Threshold: {threshold_value} ->", energy > threshold_value)
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
    torch = get_torch()
    if model_instance is not None:
        try: del model_instance
        except Exception as e: print(f"Error deleting whisper model: {e}")
    if classifier_instance is not None:
        try: del classifier_instance
        except Exception as e: print(f"Error deleting classifier: {e}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Models deleted, CUDA cache cleared.")

def perform_auto_search(query, worker_thread, score=None, max_len=10):
    """Executes a Google Custom Search and emits results to the main thread."""
    if not query or not worker_thread:
        return

    query_full = query.strip() + ' KJV Bible verse'
    url = 'https://customsearch.googleapis.com/customsearch/v1?'
    params = {
        "key": os.getenv('API_KEY'),
        "cx": os.getenv('SEARCH_ENGINE_ID'),
        "q": query_full + ' KJV Bible verse',
        "safe": "active", 'hl': 'en', 'num': 10,
    }
    
    try:
        response = httpx.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        results = response.json()
        items = results.get("items", [])
        print(f"Auto-search found {len(items)} items for query: '{query_full}'")
        
        if items and worker_thread:
             # This signal is defined in TranscriptionWorker in your other file
             worker_thread.autoSearchResults.emit(items, query_full, score, max_len)
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

# ===================================================================
# CONSUMER THREAD (as a QThread)
# ===================================================================

class ProcessorThread(QThread):
    """
    This is the "Consumer" thread. It inherits from QThread.
    It does all the heavy lifting (VAD, Transcribing, Searching)
    so the audio thread never blocks.
    """
    textReady = pyqtSignal(str)  # This is the thread-safe signal for the GUI

    def __init__(self, audio_queue, controller, worker_thread):
        super().__init__()
        torch = get_torch()
        max_trys = 5
        for _ in range(max_trys):
            pipeline = get_pipeline()
            if pipeline is not None: break
        WhisperModel = get_WhisperModel()
        self.audio_queue = audio_queue
        self.controller = controller
        self.worker_thread = worker_thread # To emit autoSearchResults
        
        # --- Load all settings ---
        print("Processor: Loading settings...")
        self.beam_size = int(settings.value('beam') or 5)
        self.best_of = int(settings.value('best') or 2)
        self.temperature = float(settings.value('temperature') or 0.00)
        self.language = settings.value('language') or 'en'
        self.energy_threshold = float(settings.value('energy') or 0.10)
        print(f"Processor: Energy threshold set to {settings.value('energy')}")
        model_size = (settings.value('model') or 'tiny').lower()
        cpu_cores = int(settings.value('cores') or 1)
        processing = settings.value('processing') or ('CPU' if not torch.cuda.is_available() else 'GPU')
        
        self.RATE = int(settings.value('rate') or 16000)
        self.CHUNK = int(settings.value('chunks') or 1024)
        self.CHANNELS = int(settings.value('channel') or 1)
        self.FORMAT = pyaudio.paInt16
        # Get sample width from a throwaway instance
        self.SAMPLE_WIDTH = pyaudio.PyAudio().get_sample_size(self.FORMAT)
        
        self.silence_length = float(settings.value('silence') or 0.5)
        self.min_record_len = float(settings.value('minlen') or .25)
        self.max_record_len = float(settings.value('maxlen') or 1)
        computation_type = "float16" if processing == "GPU" else "int8"
        self.auto_search_size = int(settings.value('auto_length') or 1)
        self.confidence_threshold = float(percent_to_log_prob(settings.value('confidence_threshold')) or -0.4)
        self.use_prev_context = int(settings.value('prev_context') or 0)
        
        #setting the models max and min lenghts
        # self.max_record_len = 1 
        # self.min_record_len = .5
        print(f'min len set to: {self.min_record_len}, max len set to: {self.max_record_len}')
        

        # --- Load all models ---
        self.model = None
        self.bible_classifier_model = None
        
        print("Processor: Loading models...")
        model_source = 'finetuned_distilbert_bert'
        try:
            self.bible_classifier_model = pipeline(
                'text-classification',
                model=resource_path(model_source),
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"Processor: Classifier loaded from {model_source}")
        except Exception as e:
            print(f"Processor: Failed to load classifier: {e}")

        device_type = 'cuda' if processing == "GPU" and torch.cuda.is_available() else 'cpu'
        if processing == "GPU" and device_type == 'cpu':
             print("Processor: GPU requested but not available, falling back to CPU int8")
             computation_type = "int8"

        self.model = WhisperModel(
            model_size,
            device=device_type,
            compute_type=computation_type,
            num_workers=cpu_cores,
            download_root=resource_path(os.path.join(f'./models/{model_size}'))
        )
        

        print("Processor: Whisper model loaded.")

    def run(self):
        """Main processing loop for the consumer thread."""
        print("Processor: Thread started.")
        prev_context = ""
        counter = 1
        silence_frames_threshold = int(self.silence_length * self.RATE / self.CHUNK)
        
        frames = []
        silent_frames = 0
        is_recording_speech = False
        torch = get_torch()
        
        try:
            while not self.controller.is_stopped():
                try:
                    data = self.audio_queue.get(timeout=1.0)
                    
                    is_loud = get_energy_threshold(data, threshold_value=self.energy_threshold)

                    if not is_recording_speech:
                        if is_loud:
                            print("Processor: Speech detected, starting segment...")
                            is_recording_speech = True
                            frames.append(data)
                            silent_frames = 0
                    else:
                        frames.append(data)
                        if is_loud:
                            silent_frames = 0
                        else:
                            silent_frames += 1
                        
                        recording_length_sec = (len(frames) * self.CHUNK) / self.RATE
                        segment_finished = False
                        
                        if silent_frames >= silence_frames_threshold:
                            if recording_length_sec >= self.min_record_len:
                                segment_finished = True
                        
                        if recording_length_sec >= self.max_record_len:
                            segment_finished = True
                        
                        # --- Process This Segment ---
                        if segment_finished:
                            print(f"Processor: Segment {counter} finished. Processing...")
                            
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                temp_filename = temp_file.name
                            
                            try:
                                with wave.open(temp_filename, 'wb') as wf:
                                    wf.setnchannels(self.CHANNELS)
                                    wf.setsampwidth(self.SAMPLE_WIDTH)
                                    wf.setframerate(self.RATE)
                                    wf.writeframes(b''.join(frames))
                                # Transcribe from file
                                # Replace the entire segment processing section (around lines 195-235)

                                # After writing the WAV file and before transcription:
                                segments = transcrip(
                                    model=self.model,
                                    filename=temp_filename,
                                    beam_size=self.beam_size,
                                    best_of=self.best_of,
                                    temperature=self.temperature,
                                    language=self.language,
                                    vad_filter=True,
                                    word_timestamps=False,
                                    prev_segment_text=prev_context if self.use_prev_context else "",
                                    log_prob_threshold=self.confidence_threshold
                                )

                                full_trans = ""
                                current_segment_text = ""  # NEW: Build context from current segment

                                for segment in segments:
                                    print(f'Processor: {segment.text.strip()} (avg_logprob: {segment.avg_logprob})')
                                    full_trans += segment.text.strip() + " "
                                    
                                    # FIXED: Build context regardless of confidence
                                    # Only use confidence threshold for Whisper's initial_prompt
                                    current_segment_text += " " + segment.text

                                current_text_lower = full_trans.lower()
                                print(f"Processor: Full transcription: '{current_text_lower.strip()}'")

                                if full_trans.strip():
                                    # Emit signal for GUI
                                    self.textReady.emit(full_trans)

                                    if self.bible_classifier_model:
                                        # Classify ONLY the current segment (no contamination)
                                        print(f"Processor: prev_context='{prev_context}'")
                                        print(f"Processor: current_segment_text='{current_segment_text.strip()}'")
                                        print(f"Processor: Classifying CURRENT segment only: '{current_segment_text.strip()}'")
                                        
                                        result = self.bible_classifier_model(current_segment_text.strip())
                                        label = result[0]['label']
                                        score = result[0]['score']
                                        print(f"Processor: Classified as: {label} ({score:.2f})")
                                        
                                        if label == 'bible' and self.worker_thread:
                                            # Search with FULL context (prev + current) for continuity
                                            search_text = (prev_context + " " + current_segment_text).strip()
                                            
                                            #Limit search length to avoid bloat
                                            words = search_text.split()
                                            if len(words) > 20:
                                                search_text = " ".join(words[-20:])  # Last 20 words
                                            
                                            print(f"Processor: Searching with combined context: '{search_text}'")
                                            perform_auto_search(
                                                query=search_text,
                                                worker_thread=self.worker_thread,
                                                score=score,
                                                max_len=self.auto_search_size
                                            )
                                            
                                            prev_context = current_segment_text.strip()
                                        else:
                                            print(f"Processor: Not Bible, resetting context")
                                            prev_context = ""
                                    

                                del segments
                                del full_trans

                            except Exception as e:
                                print(f"Processor: Error during segment processing: {e}")
                            finally:
                                if os.path.exists(temp_filename):
                                    os.remove(temp_filename)

                            # Reset for next segment
                            frames = []
                            silent_frames = 0
                            is_recording_speech = False
                            
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                except Empty:
                    continue
                except Exception as e:
                    print(f"Processor: Error in inner loop: {e}")
                    frames = []
                    silent_frames = 0
                    is_recording_speech = False
        
        except Exception as e:
            print(f"Processor: Critical error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Processor: Shutting down...")
            cleanup_model_resources(self.model, self.bible_classifier_model)
            print("Processor: Thread finished.")

# ===================================================================
# PRODUCER FUNCTION (run_transcription)
# ===================================================================
def transcrip(model, filename, beam_size, best_of, temperature, language, vad_filter, word_timestamps, prev_segment_text = "", log_prob_threshold=-0.4):
    context = prev_segment_text[-400:] if prev_segment_text else ""
    print('here is the prev segment text:', prev_segment_text)
    segments, _ = model.transcribe(
                                    filename,
                                    beam_size=beam_size,
                                    best_of=best_of,
                                    temperature=temperature,
                                    language=language,
                                    initial_prompt=context, #the last 400 chars
                                    vad_filter=vad_filter, 
                                    word_timestamps=word_timestamps,
                                    log_prob_threshold=log_prob_threshold
                                )
    return segments

def run_transcription(recording_page, search_Page=None, lineEdit=None, worker_thread=None, controller=None):
    """
    This is the "Producer" function.
    It runs inside the TranscriptionWorker QThread.
    It starts the ProcessorThread and then only reads from PyAudio.
    """
    if controller is None:
        controller = TranscriptionController()

    load_dotenv(os.path.join(basedir, '.env'))

    CHANNELS = int(settings.value('channel') or 1)
    RATE = int(settings.value('rate') or 16000)
    CHUNK = int(settings.value('chunks') or 1024)
    FORMAT = pyaudio.paInt16
    
    audio = None
    stream = None
    processor_thread = None
    
    try:
        audio_queue = Queue(maxsize=10) # Max 10 chunks in queue
        
        # Create the Consumer thread
        processor_thread = ProcessorThread(
            audio_queue=audio_queue,
            controller=controller,
            worker_thread=worker_thread # Pass the main TranscriptionWorker
        )
        
        # --- FIX: Connect Signals BEFORE starting the thread ---
        if hasattr(worker_thread, 'guiTextReady'):
            processor_thread.textReady.connect(worker_thread.guiTextReady)
            print("Audio (Producer): Connected ProcessorThread.textReady to worker_thread.guiTextReady")
        else:
            print("CRITICAL: worker_thread does not have 'guiTextReady' signal! GUI text will not update.")

        # NOW start the processor thread
        processor_thread.start()
        print("Audio (Producer): ProcessorThread started")

        def initialize_audio_stream():
            nonlocal audio, stream
            cleanup_audio_resources(audio, stream)
            audio, stream = None, None
            if controller.is_stopped(): 
                return False
            
            audio_inst = pyaudio.PyAudio()
            stream_inst = None
            max_retries = 5
            
            for i in range(max_retries):
                if controller.is_stopped(): 
                    if audio_inst:
                        audio_inst.terminate()
                    return False
                try:
                    stream_inst = audio_inst.open(
                        format=FORMAT, 
                        channels=CHANNELS, 
                        rate=RATE, 
                        input=True, 
                        frames_per_buffer=CHUNK
                    )
                    print("Audio (Producer): Stream initialized.")
                    audio, stream = audio_inst, stream_inst
                    return True
                except Exception as e:
                    print(f"Audio (Producer): Init failed (attempt {i+1}/{max_retries}): {e}")
                    time.sleep(2)
            
            if audio_inst: 
                audio_inst.terminate()
            return False

        if not initialize_audio_stream():
            print("Audio (Producer): Could not initialize. Exiting thread.")
            return

        print("Audio (Producer): Recording...")
        while not controller.is_stopped():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                # print("Audio (Producer): Read audio chunk.",)
                
                # Use a non-blocking put with try/except
                try:
                    audio_queue.put(data, block=True, timeout=0.5)
                except Exception as queue_error:
                    # Queue full or other error - skip this chunk
                    print('here is the queue error:', queue_error)
                    pass
                    
            except (OSError, IOError) as e:
                print(f"Audio (Producer): Stream error: {e}")
                print("Audio (Producer): Attempting to reinitialize...")
                if not initialize_audio_stream():
                    print("Audio (Producer): Failed to recover. Exiting.")
                    break
                
    except Exception as e:
        print(f"Audio (Producer): Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Audio (Producer): Shutting down...")
        controller.stop()
        
        if processor_thread is not None:
            print("Audio (Producer): Waiting for processor to finish...")
            processor_thread.wait(5000) # Wait 5 seconds
            if processor_thread.isRunning():
                print("Audio (Producer): Processor didn't stop, terminating...")
                processor_thread.terminate()
                processor_thread.wait()
        
        cleanup_audio_resources(audio, stream)
        print("Audio (Producer): Thread completely finished.")