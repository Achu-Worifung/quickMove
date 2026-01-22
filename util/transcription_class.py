from widgets.SearchWidget import QThread, pyqtSignal
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np 
import sounddevice as sd
import sys
import queue
import torch 
import time 
import os 
from util.util import resource_path
import threading
from PyQt5.QtCore import QSettings


class AudioProcessor:
    """Simple audio processor for normalization"""
    
    @staticmethod
    def normalize_audio(audio_chunk):
        """
        Normalize audio to prevent clipping and ensure consistent volume.
        
        :param audio_chunk: Audio data as numpy array
        :return: Normalized audio chunk
        """
        if len(audio_chunk) == 0:
            return audio_chunk
            
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 0.01:
            return audio_chunk * (0.9 / max_val)
        return audio_chunk


class ContextManager:
    """Thread-safe context management for transcription continuity"""
    
    def __init__(self, max_context_chars=300):
        self.max_context_chars = max_context_chars
        self.context_buffer = []
        self.lock = threading.Lock()  # Thread safety
        
    def add_text(self, text):
        """Add text to context buffer (thread-safe)"""
        if text and len(text.strip()) > 2:
            with self.lock:
                self.context_buffer.append(text.strip())
                # Keep only last 3 transcriptions
                if len(self.context_buffer) > 3:
                    self.context_buffer.pop(0)
    
    def get_context(self):
        """Get context string (thread-safe)"""
        with self.lock:
            if not self.context_buffer:
                return None
            
            # Join last transcriptions
            context = " ".join(self.context_buffer[-2:])
            
            # Truncate if too long
            if len(context) > self.max_context_chars:
                return "..." + context[-self.max_context_chars:]
            
            return context


class TranscriptionWorker(QThread):
    
    finished = pyqtSignal()
    autoSearchResults = pyqtSignal(list, str, float, int)  # Emit results, query, confidence, and max_len
    guitextReady = pyqtSignal(str)  # emit transcribed text 
    loadingStatus = pyqtSignal()  # signal to update loading status
    statusUpdate = pyqtSignal(str)  # New: emit status updates (listening, processing, etc.)
    
    def __init__(self, parent=None, search_page=None):
        super().__init__(parent)
        self.record_page = parent
        self.search_page = search_page
        self.lineEdit = self.record_page.lineEdit
        self.settings = QSettings("MyApp", "AutomataSimulator")
        
        # Initialize audio processor and context manager
        self.audio_processor = AudioProcessor()
        self.context_manager = ContextManager()
        
        # Reading data from settings 
        self.beam_size = 2  # REAL-TIME SAFE
        self.best_of = 2
        self.temperature = 0.0

        self.language = self.settings.value('language') or 'en'
        self.energy_threshold = float(self.settings.value('energy') or 0.10)
        self.model_size = (self.settings.value('model') or 'tiny').lower()
        self.cpu_cores = int(self.settings.value('cores') or 1)
        self.confidence_threshold = float(self.settings.value('confidence_threshold') or -0.4) or -0.4
        self.auto_search_size = int(self.settings.value('auto_length') or 1)

        self.RATE = int(self.settings.value('rate') or 16000)
        self.CHUNK = int(self.settings.value('chunks') or 1024)
        self.CHANNELS = int(self.settings.value('channel') or 1)

        # Transcription parameters (based on reference implementation)
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * 0.5)  # 0.5-second chunks
        self.vad_threshold = 0.7
        self.min_speech_duration = 0.2
        self.min_silence_duration = 0.2
        
        # Control parameters
        self.max_accumulated_duration = 1.5  # Transcribe every 1.5 seconds of speech
        self.min_transcription_audio = 0.8   # Minimum 0.8 seconds of audio to transcribe

        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.computation_type = "float16" if self.device_type == 'cuda' else "int8"

        # Initialize global variables
        self.whisper, self.vad = self.load_transcription_model()

        # IMPROVEMENT 1: Use maxsize to prevent memory issues from unbounded queues
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size
        self.transcription_queue = queue.Queue(maxsize=5)  # Limit pending transcriptions
        
        # IMPROVEMENT 2: Thread-safe state management
        self.state_lock = threading.Lock()
        self.accumulated_audio = []
        self.is_speaking = False
        self.silence_counter = 0
        
        # Thread control
        self.running = True
        
        # Performance monitoring
        self.stats = {
            'audio_drops': 0,
            'transcription_count': 0,
            'avg_transcription_time': 0.0
        }
        
        # Emit loading status complete
        self.loadingStatus.emit()
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        This method is called (from a separate thread) for each audio block.
        Runs in audio thread - must be fast and non-blocking!
        """
        if status:
            if status.input_overflow:
                self.stats['audio_drops'] += 1
        
        audio_chunk = indata.copy().flatten()
        audio_chunk = self.audio_processor.normalize_audio(audio_chunk)
        audio_chunk = audio_chunk.astype(np.float32)
        
        # Performance monitoring: Non-blocking queue put with error handling
        try:
            self.audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            # Queue is full - skip this chunk to avoid blocking audio thread
            self.stats['audio_drops'] += 1
        
    def check_for_speech(self, audio_chunk):
        """
        Check for speech in the provided audio chunk using the VAD model.
        """
        if self.vad is None or len(audio_chunk) < self.sample_rate * 0.1:
            return True  # Assume speech is present if VAD is not loaded or audio is too short
        try:
            timestamps = get_speech_timestamps(
                audio_chunk,
                self.vad,
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_speech_duration_ms=int(self.min_speech_duration * 1000),
                min_silence_duration_ms=50,
                return_seconds=False
            )
            return len(timestamps) > 0
        except Exception as e:
            print(f"Error in check_for_speech: {e}")
            return True
    
    def process_audio_chunk(self, chunk):
        """
        Process single audio chunk - runs in audio processing thread
        This method decides WHEN to transcribe but doesn't do the transcription itself
        """
        # Check for speech
        has_speech = self.check_for_speech(chunk)
        
        # Thread-safe state access
        with self.state_lock:
            if has_speech:
                self.accumulated_audio.append(chunk)
                
                if not self.is_speaking:
                    self.is_speaking = True
                    self.statusUpdate.emit("speaking")
                
                self.silence_counter = 0
                
                # Check if we should transcribe
                total_duration = sum(len(a) for a in self.accumulated_audio) / self.sample_rate
                
                # Transcribe frequently for snappy feel
                if total_duration >= self.max_accumulated_duration and self.is_speaking:
                    # Get context
                    context = self.context_manager.get_context()
                    
                    # Use recent audio (last 4 chunks = 2 seconds)
                    audio_to_transcribe = np.concatenate(self.accumulated_audio[-4:]) \
                        if len(self.accumulated_audio) >= 4 else np.concatenate(self.accumulated_audio)
                    
                    #Non-blocking queue with drop on full
                    try:
                        self.transcription_queue.put_nowait({
                            'audio': audio_to_transcribe.copy(),
                            'context': context,
                            'type': 'partial'
                        })
                        self.statusUpdate.emit("processing")
                    except queue.Full:
                        print("⚠️ Transcription queue full, skipping transcription")
                    
                    # Keep only last 1 chunk for overlap
                    if len(self.accumulated_audio) > 1:
                        self.accumulated_audio = self.accumulated_audio[-1:]
                        
            elif self.is_speaking:
                # We were speaking, now in silence
                self.silence_counter += 1
                
                # Check if silence is long enough to finalize
                silence_time = self.silence_counter * (self.chunk_size / self.sample_rate)
                
                if silence_time >= self.min_silence_duration:
                    if self.accumulated_audio:
                        # Only transcribe if we have enough audio
                        total_duration = sum(len(a) for a in self.accumulated_audio) / self.sample_rate
                        
                        if total_duration >= self.min_transcription_audio:
                            # Get context
                            context = self.context_manager.get_context()
                            
                            # Queue transcription task for remaining audio
                            audio_to_transcribe = np.concatenate(self.accumulated_audio)
                            
                            try:
                                self.transcription_queue.put_nowait({
                                    'audio': audio_to_transcribe.copy(),
                                    'context': context,
                                    'type': 'final'
                                })
                                self.statusUpdate.emit("processing")
                            except queue.Full:
                                print("⚠️ Transcription queue full, skipping final transcription")
                    
                    # Reset for next utterance
                    self.accumulated_audio = []
                    self.is_speaking = False
                    self.silence_counter = 0
                    self.statusUpdate.emit("listening")
    
    def transcribe_audio_chunk(self, audio_data, context=None):
        """
        Fast transcription with minimal parameters
        Runs in transcription thread - can be slow
        """
        if len(audio_data) < self.sample_rate * 0.3:  # Less than 0.3s
            return ""
        
        # Track transcription time
        start_time = time.time()
        
        try:
            # Use minimal parameters for speed
            segments, _ = self.whisper.transcribe(
                audio_data,
                language="en",
                beam_size=self.beam_size,
                best_of=self.best_of,
                temperature=self.temperature,
                condition_on_previous_text=context is not None,
                initial_prompt=context,
                vad_filter=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                word_timestamps=False,
                no_speech_threshold=0.6,
                suppress_blank=True,
                without_timestamps=True,
                prepend_punctuations="\"'¿([{",
                append_punctuations="\"'.。,，!！?？:：)]}、",
            )
            
            # Collect text
            texts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    texts.append(text)
            
            # Update performance stats
            elapsed = time.time() - start_time
            self.stats['transcription_count'] += 1
            self.stats['avg_transcription_time'] = (
                (self.stats['avg_transcription_time'] * (self.stats['transcription_count'] - 1) + elapsed) 
                / self.stats['transcription_count']
            )
            
            if not texts:
                return ""
            
            result = " ".join(texts).strip()
            
            # Log performance warnings
            if elapsed > 2.0:
                print(f"⚠️ Slow transcription: {elapsed:.2f}s for {len(audio_data)/self.sample_rate:.2f}s audio")
            
            return result
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def load_transcription_model(self, num_workers: int = 2, cpu_threads: int = 4):
        """
        Load the transcription model and Voice Activity Detection (VAD) model.
        """
        try:
            self.whisper = WhisperModel(
                self.model_size,
                device=self.device_type,
                compute_type=self.computation_type,
                num_workers=num_workers,
                cpu_threads=cpu_threads if self.device_type == "cpu" else 0,
                download_root=resource_path(os.path.join("models", f"{self.model_size}")),
            )
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.whisper = None

        try:
            self.vad = load_silero_vad()
            print("VAD model loaded successfully.")
        except Exception as e:
            print(f"Error loading VAD model: {e}")
            self.vad = None
            
        return self.whisper, self.vad

    def record_audio(self):
        """
        Record audio chunks - runs in its own thread
        """
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                latency='low'
            ):
                print("Audio recording started")
                while self.running:
                    time.sleep(0.1)  # Keep the stream alive
        except Exception as e:
            print(f"Error in recording thread: {e}")

    def process_audio(self):
        """
        Process audio chunks - runs in its own thread
        This thread is fast and handles VAD and audio accumulation
        """
        print("Audio processing started")
        try:
            while self.running:
                try:
                    # Get next audio chunk with timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    self.process_audio_chunk(chunk)
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error in audio processing thread: {e}")

    def transcribe_loop(self):
        """
        Transcription loop - runs in its own thread
        This thread handles the slow transcription work
        """
        print("Transcription loop started")
        try:
            while self.running:
                try:
                    # Get next transcription task with timeout
                    task = self.transcription_queue.get(timeout=0.1)
                    
                    audio_data = task['audio']
                    context = task['context']
                    task_type = task['type']
                    
                    # Do the actual transcription (this is slow)
                    text = self.transcribe_audio_chunk(audio_data, context)
                    
                    if text:
                        # Add to context
                        self.context_manager.add_text(text)
                        
                        # Emit the transcribed text
                        self.guitextReady.emit(text)
                        
                        # IMPROVEMENT 9: Clear queue if we're falling behind
                        if self.transcription_queue.qsize() > 3:
                            print(f"⚠️ Transcription backlog: {self.transcription_queue.qsize()} tasks")
                        
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error in transcription loop: {e}")

    def stop(self):
        """
        IMPROVEMENT 10: Graceful shutdown method
        """
        print("Stopping transcription worker...")
        self.running = False
        
        # Print final stats
        print("\n=== Transcription Statistics ===")
        print(f"Total transcriptions: {self.stats['transcription_count']}")
        print(f"Average transcription time: {self.stats['avg_transcription_time']:.3f}s")
        print(f"Audio drops: {self.stats['audio_drops']}")
        print("================================\n")

    def run(self):
        """
        Main thread loop - coordinates the three worker threads
        """
        print("Starting transcription worker...")
        
        try:
            # Create three threads:
            # 1. Audio recording (fast, handles audio callback)
            # 2. Audio processing (fast, handles VAD and accumulation)
            # 3. Transcription (slow, handles Whisper transcription)
            
            recording_thread = threading.Thread(target=self.record_audio, daemon=True, name="AudioRecording")
            processing_thread = threading.Thread(target=self.process_audio, daemon=True, name="AudioProcessing")
            transcription_thread = threading.Thread(target=self.transcribe_loop, daemon=True, name="Transcription")
            
            # Start all threads
            recording_thread.start()
            processing_thread.start()
            transcription_thread.start()
            
            # Wait for threads to complete
            recording_thread.join()
            processing_thread.join()
            transcription_thread.join()
            
        except Exception as e:
            print(f"Error in transcription worker: {e}")
        finally:
            self.running = False
            self.finished.emit()


