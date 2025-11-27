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


def cleanup_model_resources(model_instance, classifier_instance, text_ready_signal=None, worker_thread=None):
    pass

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
        self._is_processing = False  # New flag to track processing state
        self._lock = threading.Lock()
    
    def stop(self):
        with self._lock:
            self._should_stop = True
    
    def is_stopped(self):
        with self._lock:
            return self._should_stop

    def set_processing(self, value):
        """Set the processing state."""
        with self._lock:
            self._is_processing = value

    def is_processing(self):
        """Check if processing is active."""
        with self._lock:
            return self._is_processing

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
                

        # --- Load all models ---
        self.model = None
        self.bible_classifier_model = None
        
        model_source = 'finetuned_distilbert_bert'
        try:
            self.bible_classifier_model = pipeline(
                'text-classification',
                model=resource_path(model_source),
                device=0 if torch.cuda.is_available() else -1
            )
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
                    if data is None:
                        print("Processor: Received stop signal (Poison Pill). Exiting loop.")
                        break
                    
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
            cleanup_model_resources(self.model, self.bible_classifier_model, self.textReady, self.worker_thread)
            print("Processor: Thread finished.")

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


# ===================================================================
# PRODUCER FUNCTION (run_transcription)
# ===================================================================

def run_transcription(recording_page, search_Page=None, lineEdit=None, worker_thread=None, controller=None):
    if controller is None:
        controller = TranscriptionController()

    CHANNELS = int(settings.value('channel') or 1)
    RATE = int(settings.value('rate') or 16000)
    CHUNK = int(settings.value('chunks') or 1024)
    FORMAT = pyaudio.paInt16

    audio_queue = Queue(maxsize=10)
    audio = None
    stream = None

    # Start processor thread first
    processor_thread = ProcessorThread(audio_queue, controller, worker_thread)
    processor_thread.textReady.connect(worker_thread.guiTextReady)
    processor_thread.start()

    print("Audio (Producer): ProcessorThread started")

   
    def callback(in_data, frame_count, time_info, status):
        if controller.is_stopped():
            return (None, pyaudio.paComplete)

        try:
            audio_queue.put_nowait(in_data)
        except:
            # Queue full → drop audio silently
            pass

        return (None, pyaudio.paContinue)

    try:
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=callback,
        )

        stream.start_stream()
        print("Audio (Producer): Callback stream started")

        # Keep thread alive while callback runs
        while stream.is_active() and not controller.is_stopped():
            QThread.msleep(50)

    except Exception as e:
        print("Audio (Producer): Error:", e)

    finally:
        print("Audio (Producer): Cleaning up callback stream...")
        processor_thread.controller.stop()
        
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
                print('the stream is closed')
            except:
                pass

        if audio is not None:
            try:
                audio.terminate()
                print('the audio is terminated')
            except:
                pass

        # Send poison pill
        try:
            audio_queue.put_nowait(None)
            print("Audio (Producer): Poison pill sent.")
        except:
            pass
        
        processor_thread.wait()
        processor_thread.deleteLater()
        print("Audio (Producer): Fully stopped.")
