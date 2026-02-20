import torch 
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForSequenceClassification
from widgets.SearchWidget import QThread, pyqtSignal
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np 
import sounddevice as sd
import queue
import time 
import os 
from util.util import resource_path
import threading
from PyQt5.QtCore import QSettings
import re
from rank_bm25 import BM25Okapi
from scipy.special import expit
import unicodedata
from util.classifier import Classifier
from util.biblesearch import BibleSearch
from rapidfuzz import fuzz
from util.extract_verse import extract_bible_reference

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
        self.classifier = None
        self.transcribed_chunks = [] #stores 2 transcribed chunks for context
        self.classification_chunks = [] #stores the classification
        self.verse_buffer = {} #stores recent verses for context in search
        
        self._pause = False
        self.pause_event = threading.Event()
        self.pause_event.set()
        self._stop = False
        self.lineEdit = self.record_page.lineEdit
        self.settings = QSettings("MyApp", "AutomataSimulator")
        
        # Initialize audio processor and context manager
        self.audio_processor = AudioProcessor()
        self.context_manager = ContextManager()
        
        # Initialize Bible search
        self.bible_search = None
        
        # Read transcription model settings from QSettings
        self.beam_size = int(self.settings.value('beam') or 3)
        self.best_of = int(self.settings.value('best') or 3)
        self.temperature = float(self.settings.value('temperature') or 0.0)
        self.language = self.settings.value('language') or 'en'
        self.energy_threshold = float(self.settings.value('energy') or 0.001)
        self.model_size = (self.settings.value('model') or "").lower()
        self.cpu_cores = int(self.settings.value('cores') or 4)
        self.confidence_threshold = float(self.settings.value('transcription_confidence') or -1.0)
        self.auto_search_size = int(self.settings.value('auto_length') or 4)

        # Audio recording settings
        self.RATE = int(self.settings.value('rate') or 16000)
        self.CHUNK = int(self.settings.value('chunks') or 1024)
        self.CHANNELS = int(self.settings.value('channel') or 1)
        
        # Search settings (with fallback defaults if not in settings)
        self.semantic_topk = int(self.settings.value('semantic_topk') or 100)
        self.auto_topk = int(self.settings.value('auto_topk') or 10)
        self.BM25_TOP_K = int(self.settings.value('bm25_top_k') or 1000)

        # Voice Activity Detection and transcription parameters
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * 0.5)  # 0.5-second chunks
        self.vad_threshold = float(self.settings.value('vad_threshold') or 0.7)
        self.min_speech_duration = float(self.settings.value('minlen') or 0.2)
        self.min_silence_duration = float(self.settings.value('silence') or 0.3)
        
        # Control parameters for continuous transcription
        self.max_accumulated_duration = float(self.settings.value('maxlen') or 1.5)
        self.min_transcription_audio = float(self.settings.value('min_transcription_audio') or 0.8)

        # Determine device and computation type from settings
        processing_mode = self.settings.value('processing') or 'CPU'
        self.device_type = 'cuda' if torch.cuda.is_available() and processing_mode.upper() == 'GPU' else 'cpu'
        self.computation_type = "float16" if self.device_type == 'cuda' else "int8"
        
        # Mark that models haven't been loaded yet
        self.models_loaded = False
       
        
        # Initialize global variables to None (will be loaded in run())
        self.whisper = None
        self.vad = None

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
        self.prev_transcription = ""
        
        # Performance monitoring
        self.stats = {
            'audio_drops': 0,
            'transcription_count': 0,
            'avg_transcription_time': 0.0
        }
        

    def stop(self):
        self._stop = True
        print('stopping transcription thread...')
    
    def initialize_bible_search(self):
        """Initialize Bible search in background thread"""
        try:
            print("Initializing Bible search engine...")
            self.bible_search = BibleSearch(device_type=self.device_type)
            print("Bible search engine ready!")
        except Exception as e:
            print(f"Failed to initialize Bible search: {e}")
            self.bible_search = None
    
    def load_transcription_model(self, num_workers: int = 2):
        """
        Load the transcription model and Voice Activity Detection (VAD) model.
        Uses cpu_threads from settings if using CPU device.
        """
        try:
            cpu_threads = self.cpu_cores if self.device_type == "cpu" else 0
            print(f"Loading Whisper model: {self.model_size}")
            print(f"  Device: {self.device_type}")
            print(f"  Compute type: {self.computation_type}")
            print(f"  CPU cores: {cpu_threads}")
            print(f"  Model location: {resource_path(os.path.join('models', f'{self.model_size}'))}")
            
            self.whisper = WhisperModel(
                self.model_size,
                device=self.device_type,
                compute_type=self.computation_type,
                num_workers=num_workers,
                cpu_threads=cpu_threads,
                download_root=resource_path(os.path.join("models", f"{self.model_size}")),
            )
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            import traceback
            traceback.print_exc()
            self.whisper = None

        try:
            self.vad = load_silero_vad()
            print("VAD model loaded successfully.")
        except Exception as e:
            print(f"Error loading VAD model: {e}")
            self.vad = None
            
        return self.whisper, self.vad

    def _clear_queue(self, q: queue.Queue):
        """
        Safely drain a queue.Queue instance without relying on module-level functions.
        """
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return
    
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
                        # Clear pending items to recover from backlog
                        # Drain the transcription queue to recover from backlog
                        self._clear_queue(self.transcription_queue)
                        print("Transcription queue full, cleared backlog")
                    
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
                                self._clear_queue(self.transcription_queue)
                                print("Transcription queue full, skipping final transcription")
                    
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
        if self.whisper is None:
            print("Whisper model not loaded. Please check model loading errors above.")
            return ""
            
        if len(audio_data) < self.sample_rate * 0.3:  # Less than 0.3s
            return ""
        
        # Track transcription time
        
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
    
            
            if not texts:
                print("No text transcribed: transcription confidence may be lower than expected.")
                return ""
            
            result = " ".join(texts).strip()
            
            # print(f"Transcribed: {result} confidence {segments[0].confidence if segments else 0}")
            
           
            
            return result
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
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
                # Inform UI that we're ready and listening
                self.statusUpdate.emit("listening")
                while self.running and not self._stop:
                    with self.state_lock:
                        should_stop = self._stop
                    if should_stop:
                        break
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
            while self.running and not self._stop:
                # Emit paused status and skip processing while paused
                if self._pause:
                    self.statusUpdate.emit("paused")
                    time.sleep(0.1)
                    continue
                with self.state_lock:
                    should_stop = self._stop
                    if should_stop:
                        break
                try:
                    # Get next audio chunk with timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    self.process_audio_chunk(chunk)
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error in audio processing thread: {e}")

    def perform_auto_search(self, query):
        """Perform Bible search with the transcribed query"""
        if not self.bible_search:
            print("Bible search not initialized yet")
            return []
        
        try:
            # Perform the search using the BibleSearch class
            # Increase final_top_k to get multiple translations per verse
            results = self.bible_search.search(
                query=query,
                bm25_top_k=self.BM25_TOP_K,
                semantic_top_k=self.semantic_topk,
                final_top_k=self.auto_topk * 2  # Allow multiple translations per verse
            )
            
            # Sort results by score in DESCENDING order (highest first)
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Format results for the UI with Bible version
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {"reference": result['ref'],
                    "text": result['text'],
                    "score": f"{result['score']:.2f}%",
                    "book": result['book'],
                    "chapter": result['chapter'],
                    "verse": result['verse'],
                    "version": result['version']}
                    
                )
            
            return formatted_results
            
        except Exception as e:
            print(f" Error in auto-search: {e}")
            return ["Search error occurred. Please try again."]
   

    def normalize_text(self, text):
        text = text.lower()
        text = text.strip()
        
        # Remove "Original: " section and everything up to "Paraphrase: " (greedy match)
        text = re.sub(r'original:.*?paraphrase:\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove standalone "Paraphrase: " prefix
        text = re.sub(r'^paraphrase:\s*', '', text, flags=re.IGNORECASE)
        
        # Remove escaped quotes
        text = text.replace('\\"', '"')
        
        # Remove outer quotes if present
        text = text.strip('"').strip("'").strip()
        
        # Handle escape sequences
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\r', ' ')
        text = text.replace('\\', '')
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        # Normalize unicode and keep only ASCII
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
        return text

    def transcribe_loop(self):
        """
        Transcription loop - runs in its own thread
        This thread handles the slow transcription work
        """
        print("Transcription loop started")
        try:
            while self.running and not self._stop:
                if self._pause:
                    time.sleep(1)
                    continue
                

                try:
                    # Get next transcription task with timeout
                    task = self.transcription_queue.get(timeout=0.1)

                    audio_data = task['audio']
                    context = task['context']
                    task_type = task['type']

                    # Transcribe chunk
                    text = self.transcribe_audio_chunk(audio_data, context)
                    if not text:
                        continue

                    # Normalize & add to context
                    text = self.normalize_text(text)
                    self.context_manager.add_text(text)

                    # ---- Buffers ----
                    self.transcribed_chunks.append(text)     # for UI overlap
                    self.classification_chunks.append(text)  # for classification/search

                    # Keep last 2 chunks for UI overlap
                    if len(self.transcribed_chunks) > 2:
                        self.transcribed_chunks = self.transcribed_chunks[-2:]

                    # Keep last 2 chunks for classification
                    if len(self.classification_chunks) > 2:
                        self.classification_chunks = self.classification_chunks[-2:]

                    # ---- Overlap verification for UI ----
                    if len(self.transcribed_chunks) < 2:
                        print("Not enough chunks for context, waiting for next...")
                        continue

                    prev_chunk, curr_chunk = self.transcribed_chunks[-2:]
                    overlap_index = -1
                    for i in range(len(prev_chunk)):
                        if curr_chunk.startswith(prev_chunk[i:]):
                            overlap_index = i
                            break

                    if overlap_index != -1:
                        merged_text = prev_chunk[:overlap_index] + curr_chunk
                        print(f"overlap made: {merged_text}")
                    else:
                        merged_text = prev_chunk + " " + curr_chunk
                        print(f"no overlap: {merged_text}")

                    # Emit verified text to ui
                    self.guitextReady.emit(merged_text)
                    
                    #check if reference exist in transcription 
                    spoken_references = extract_bible_reference(merged_text)
                    reference_dict = []
                    if spoken_references:
                        for ref in spoken_references:
                            print(f"Found spoken reference: {ref['full']} in transcription, skipping search and emitting directly")
                        
                            reference_dict.append({
                                'reference': ref['full'],
                                'book': ref.get('book', ''),
                            'chapter': ref.get('chapter', ''),
                                'verse': ref.get('verse', ''),
                                'text':"",
                                'score':1.0
                            })                            
                    
                        self.autoSearchResults.emit(reference_dict, "", 0.00, self.auto_search_size)
                        continue #skipping search entirely
                        
                            
                    # Use only the last unclassified chunk for classification to prevent bleed
                    chunk_to_classify = self.classification_chunks[-1]
                    #checking if this is a continuation of the previous chunk or a new phrase to classify
                    print(f"Chunk to classify: {chunk_to_classify}")
                    print(f"Verse buffer for context: {list(self.verse_buffer.keys())}")
                    if chunk_to_classify and self.verse_buffer:
                        best_score = 0
                        best_match_info = None

                        for key, verse_data in self.verse_buffer.items():
                            # Get text safely
                            v_text = verse_data.get('text', '') if isinstance(verse_data, dict) else str(verse_data)
                            
                            # Calculate alignment
                            alignment = fuzz.partial_ratio_alignment(merged_text.lower(), v_text.lower())
                            
                            if alignment.score > best_score:
                                best_score = alignment.score
                                best_match_info = {
                                    'data': verse_data,
                                    'text': v_text, 
                                    'start': alignment.dest_start,
                                    'end': alignment.dest_end
                                }

                        if best_score > 70:                            
                            # Use best_match_info specifically to avoid "last item in loop" bugs
                            target_text = best_match_info['text']
                            target_ref = best_match_info['data'].get('reference', 'Unknown')
                            target_book = best_match_info['data'].get('book', '')
                            target_chapter = best_match_info['data'].get('chapter', '')
                            target_verse = best_match_info['data'].get('verse', '')
                            
                            continuation_result = [{
                                'reference': target_ref,
                                'book': target_book,
                                'chapter': target_chapter,
                                'verse': target_verse,
                                'text': target_text, 
                                'score': f"{1.0:.0f}%",
                                'is_continuation': True 
                            }]
                            
                            # print(f"found continuation match: @ reference -> {target_ref} with score -> {best_score:.0f}% ")
                            
                            
                            print(f'continuation match found @ book {target_book} chapter {target_chapter} verse {target_verse} with score -> {best_score:.0f}% ')
                            
                            
                            # Emitting the "matched portion" of the verse as the query
                            matched_chunk = target_text[:best_match_info['end']]
                            
                            self.autoSearchResults.emit(continuation_result, matched_chunk, threshold, self.auto_search_size)
                            print(f'continuiting found: @ reference -> {target_ref} with score -> {best_score:.0f}% skipping the search ')
                            continue 
                        else:
                                print(f"No good continuation match found (best score: {best_score:.0f}%), for {best_match_info} \n proceeding with classification and search")
                                                                               
                    
                    results = self.classifier.classify([merged_text, chunk_to_classify])
                    label_merged, confidence_merged = results[0]
                    label_chunk, confidence_chunk = results[1]
                    
                    print(f"Classification result: merged -> {label_merged} ({confidence_merged:.2%}), chunk -> {label_chunk} ({confidence_chunk:.2%})")

                    # Keyword fallback
                    keyword_pattern = r"\b(god|lord|jesus|christ|bible|heaven|hell)\b"
                    contains_keyword = re.search(keyword_pattern, merged_text, re.IGNORECASE) is not None

                    if label_merged == "non bible" and label_chunk == "non bible" and not contains_keyword:
                        print("Non-bible and does not contain key words, skipping search")
                        #clearing the verse buffer since we likely have a new topic and the old verses won't be relevant anymore
                        self.verse_buffer = {}
                        continue  # skip search for non-Bible content
                    self.classification_chunks = []  # Clear classification buffer after use
                    # ---- Perform search ----
                    query = chunk_to_classify if label_merged == 'non bible' else merged_text #search for the chunk if the entire text is non-bible
                    results = self.perform_auto_search(query)
                    
                    

                    # Filter by threshold
                    threshold = float(self.settings.value("bible_confidence") or 60) / 100
                    filtered_results = []
                    for r in results:
                        score_str = str(r.get("score", 0)).replace("%", "")
                        #adding the verses to the verse buffer for context in future searches
                        try:
                            score_val = float(score_str)
                        except ValueError:
                            score_val = 0.0
                            print(f"Invalid score value: {score_str} in result {r['reference']}")
                        if score_val >= threshold:
                            # Store the full result dict, not just text
                            self.verse_buffer[r['reference']] = {
                                'text': self.normalize_text(r['text']),
                                'reference': r['reference'],
                                'score': score_val,
                                'version': r.get('version', ''),
                                'book': r.get('book', ''),
                                'chapter': r.get('chapter', ''),
                                'verse': r.get('verse', '')
                            }
                            filtered_results.append(r)

                    # Deduplicate highest-scoring verses
                    verse_map = {}
                    for r in filtered_results:
                        verse_key = f"{r['book'].lower()} {r['chapter']}:{r['verse']}"
                        current_score = float(str(r.get("score", 0)).replace("%", ""))
                        existing_score = float(str(verse_map.get(verse_key, {}).get("score", 0)).replace("%", ""))
                        if verse_key not in verse_map or current_score > existing_score:
                            verse_map[verse_key] = r

                    deduped_results = list(verse_map.values())
                    print(f"Search results for '{query}': {deduped_results}")

                    if deduped_results:
                        self.autoSearchResults.emit(deduped_results, query, threshold, self.auto_search_size)
                    else:
                        print("No search results found")

                    # ---- Handle transcription backlog ----
                    if self.transcription_queue.qsize() > 3:
                        print(f"Transcription backlog: {self.transcription_queue.qsize()} tasks")

                except queue.Empty:

                    continue
                    
                except Exception as e:
                    import traceback
                    print(f"Error processing transcription task: {e}")
                    traceback.print_exc()
                    # Continue to next task instead of crashing the loop
                    continue
        except Exception as e:
            import traceback
            print(f"Error in transcription loop: {e}")
            traceback.print_exc()
            
            
    def pause_transcription(self):
        """
        Toggle pause state safely. Can be called multiple times.
        Returns the new pause state (True = paused, False = running).
        """
        with self.state_lock:
            self._pause = not self._pause 
            new_value = self._pause 
        
        if new_value:
            self.pause_event.clear()
            self.statusUpdate.emit("paused")
            print('Paused transcription')
        else:
            self.pause_event.set()
            self.statusUpdate.emit("listening")
            print('Resumed transcription')
            
        return new_value
        
            
        
    def run(self):
        """
        Main thread loop - coordinates the three worker threads
        """
        print("Starting transcription worker...")
        self.classifier = Classifier()
        
         #loading the classifier model 
        self.classifier.load_classifier()
        
        #loading the search models 
        self.initialize_bible_search()
        try:
            # Load models on thread start (not in __init__)
            if not self.models_loaded:
                print("Loading Whisper and VAD models...")
                self.whisper, self.vad = self.load_transcription_model()
                self.models_loaded = True
                
                # Emit loading status complete after models are loaded
                self.loadingStatus.emit()
            
            # Create three threads:
            # 1. Audio recording (fast, handles audio callback)
            # 2. Audio processing (fast, handles VAD and accumulation)
            # 3. Transcription (slow, handles Whisper transcription)
            
            self.recording_thread = threading.Thread(target=self.record_audio, daemon=True, name="AudioRecording")
            self.processing_thread = threading.Thread(target=self.process_audio, daemon=True, name="AudioProcessing")
            self.transcription_thread = threading.Thread(target=self.transcribe_loop, daemon=True, name="Transcription")
            
            # Start all threads
            self.recording_thread.start()
            self.processing_thread.start()
            self.transcription_thread.start()
            
            # Wait for threads to complete
            self.recording_thread.join()
            self.processing_thread.join()
            self.transcription_thread.join()
            
        except Exception as e:
            print(f"Error in transcription worker: {e}")
        finally:
            self.running = False
            self._stop = True
            self.classifier.offload_classifier() # Offload classifier to free GPU memory
            self.finished.emit()