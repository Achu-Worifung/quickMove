import torch 
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForSequenceClassification
from widgets.SearchWidget import QThread, pyqtSignal
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np 
import sounddevice as sd
import sys
import queue
import time 
import os 
from util.util import resource_path
import threading
from PyQt5.QtCore import QSettings
from transformers import AutoModel, AutoTokenizer
import json
import re
import faiss
from rank_bm25 import BM25Okapi


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


class BibleSearch:
    """Search functionality for Bible verses"""
    
    def __init__(self, device_type="cpu"):
        

        self.device_type = device_type
        self.device = "cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu"
        
        # Configuration
        self.DATA_FILE = resource_path("bibles/merged_bible.json")
        self.ARTIFACT_DIR = resource_path("artifacts")
        
        # Ensure artifact directory exists
        os.makedirs(self.ARTIFACT_DIR, exist_ok=True)
        
        # File paths
        self.EMBEDDINGS_FILE = f"{self.ARTIFACT_DIR}/embeddings.npy"
        self.FAISS_FILE = f"{self.ARTIFACT_DIR}/faiss.index"
        self.METADATA_FILE = f"{self.ARTIFACT_DIR}/metadata.json"
        self.TEXTS_FILE = f"{self.ARTIFACT_DIR}/texts.json"
        
        # Model names
        self.MODEL_NAME = 'all-mpnet-base-v2'
        self.MODEL_PATH = resource_path(f"./models/search/{self.MODEL_NAME}")
        self.RE_RANKER_MODEL = 'cross-encoder_ms-marco-MiniLM-L6-v2'
        self.ONNX_MODEL_DIR = resource_path(f"./models/search/{self.RE_RANKER_MODEL.replace('/', '_')}_onnx")
        
        # Search parameters
        self.BM25_TOP_K = 5000
        self.SEMANTIC_TOP_K = 100
        self.FINAL_TOP_K = 10
        self.BATCH_SIZE = 32
        
        # Initialize components
        self.texts = None
        self.metadata = None
        self.bm25 = None
        self.embeddings = None
        self.faiss_index = None
        self.model = None
        self.tokenizer = None
        self.onnx_reranker = None
        self.reranker_tokenizer = None
        
        # Load data and models
        self.load_data_and_models()
    
    def tokenize(self, text):
        """Tokenize text for BM25"""
        return re.findall(r"\b\w+\b", text.lower())
    
    def load_data_and_models(self):
        """Load all data and models for search"""
        print("📖 Loading Bible search data...")
        
        # Load Bible data
        with open(self.DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.texts = [v["text"] for v in data]
        self.metadata = [
            {"book": v["book"], "chapter": v["chapter"], "verse": v["verse"], "version": v["version"]}
            for v in data
        ]
        
        # Build BM25 index
        print("⚙️ Building BM25 index...")
        tokenized_texts = [self.tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Load embedding model
        print(f"Using device: {self.device}")
        model_path = self.MODEL_PATH
        
        try:
            if os.path.exists(model_path):
                print(f" Loading cached embedding model from {model_path}...")
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            else:
                print(f"⬇️ Downloading embedding model {self.MODEL_NAME}...")
                self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
                os.makedirs(model_path, exist_ok=True)
                self.model.save_pretrained(model_path)
        except Exception as e:
            print(f" Failed to load embedding model: {e}")
            raise
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{self.MODEL_NAME}")
        self.model.to(self.device)
        self.model.eval()
        
        # Load ONNX reranker
        self.load_onnx_reranker()
        
        # Load or build embeddings and FAISS index
        self.load_or_build_embeddings()
        
        print("✅ Bible search initialized successfully!")
    
    def load_onnx_reranker(self):
        """Load or convert cross-encoder to ONNX format"""
        print("📥 Loading or converting ONNX reranker model...")
        if os.path.exists(self.ONNX_MODEL_DIR):
            print(f"✅ Loading ONNX reranker from {self.ONNX_MODEL_DIR}...")
            self.onnx_reranker = ORTModelForSequenceClassification.from_pretrained(
                self.ONNX_MODEL_DIR,
                provider="CPUExecutionProvider" if self.device == "cpu" else "CUDAExecutionProvider"
            )
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.ONNX_MODEL_DIR)
        else:
            print(f"⚙️ Converting {self.RE_RANKER_MODEL} to ONNX (one-time setup)...")
            self.onnx_reranker = ORTModelForSequenceClassification.from_pretrained(
                resource_path(self.RE_RANKER_MODEL),
                export=True,
                provider="CPUExecutionProvider" if self.device == "cpu" else "CUDAExecutionProvider"
            )
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(resource_path(self.RE_RANKER_MODEL))
            
            os.makedirs(self.ONNX_MODEL_DIR, exist_ok=True)
            self.onnx_reranker.save_pretrained(self.ONNX_MODEL_DIR)
            self.reranker_tokenizer.save_pretrained(self.ONNX_MODEL_DIR)
            print(f"✅ Saved ONNX model to {self.ONNX_MODEL_DIR}")
    
    def load_or_build_embeddings(self):
        print("📥 Loading or building embeddings and FAISS index...")
        """Load embeddings and FAISS index from disk or build them"""
        if os.path.exists(self.EMBEDDINGS_FILE) and os.path.exists(self.FAISS_FILE):
            print("✅ Loading embeddings and FAISS index from disk...")
            self.embeddings = np.load(self.EMBEDDINGS_FILE)
            self.faiss_index = faiss.read_index(self.FAISS_FILE)
            
            with open(self.METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            with open(self.TEXTS_FILE, "r", encoding="utf-8") as f:
                self.texts = json.load(f)
        else:
            print("⚙️ Building embeddings and FAISS index (one-time)...")
            self.embeddings = self.get_embeddings(self.texts, batch_size=self.BATCH_SIZE).astype("float32")
            
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.embeddings)
            self.faiss_index.add(self.embeddings)
            
            # Save to disk
            np.save(self.EMBEDDINGS_FILE, self.embeddings)
            faiss.write_index(self.faiss_index, self.FAISS_FILE)
            
            with open(self.METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            
            with open(self.TEXTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.texts, f)
            
            print("✅ Saved embeddings and FAISS index")
    
    def get_embeddings(self, texts, batch_size=32):
        """Get embeddings for a list of texts"""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
                emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.append(emb)
        return np.vstack(all_embs)
    
    def bm25_recall(self, query, top_k=5000):
        """BM25 keyword recall stage"""
        start = time.time()
        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        end = time.time()
        print(f"  BM25 recall: {end - start:.4f}s")
        return top_idx, scores[top_idx]
    
    def semantic_rerank(self, query, candidate_indices, top_k=100):
        """Semantic reranking stage using embeddings"""
        start = time.time()
        q_emb = self.get_embeddings([query])
        faiss.normalize_L2(q_emb)
        
        candidate_embs = self.embeddings[candidate_indices]
        faiss.normalize_L2(candidate_embs)
        
        scores = candidate_embs @ q_emb.T
        scores = scores.squeeze()
        
        top = np.argsort(scores)[::-1][:top_k]
        top_indices = candidate_indices[top]
        
        end = time.time()
        print(f"  Semantic rerank: {end - start:.4f}s")
        return top_indices

    
    def rerank_with_crossencoder(self, query, candidate_indices, top_k=10):
        """Final reranking with cross-encoder"""
        start = time.time()
        
        if len(candidate_indices) == 0:
            return []
        
        candidate_texts = [self.texts[idx] for idx in candidate_indices]
        
        # Tokenize inputs for ONNX model
        inputs = self.reranker_tokenizer(
            [query] * len(candidate_texts),
            candidate_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device if using GPU
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get scores from ONNX model
        with torch.no_grad():
            outputs = self.onnx_reranker(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        # Get top-k results
        top = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top:
            idx = candidate_indices[i]
            meta = self.metadata[idx]
            results.append({
                "score": float(scores[i]),
                "ref": f"{meta['book']} {meta['chapter']}:{meta['verse']} ({meta['version']})",
                "text": self.texts[idx],
                "book": meta["book"],
                "chapter": meta["chapter"],
                "verse": meta["verse"],
                "version": meta["version"]
            })
        
        end = time.time()
        print(f"  Cross-encoder ONNX: {end - start:.4f}s")
        
        return results
    
    def search(self, query, bm25_top_k=None, semantic_top_k=None, final_top_k=None):
        """Main search pipeline"""
        print(f"\n🔍 Searching for: '{query}'")
        start_total = time.time()
        
        # Use provided parameters or defaults
        bm25_top_k = bm25_top_k or self.BM25_TOP_K
        semantic_top_k = semantic_top_k or self.SEMANTIC_TOP_K
        final_top_k = final_top_k or self.FINAL_TOP_K
        
        # Stage 1: BM25 keyword recall
        candidate_idx, _ = self.bm25_recall(query, top_k=bm25_top_k)
        
        # Stage 2: Semantic rerank
        semantic_top_indices = self.semantic_rerank(query, candidate_idx, top_k=semantic_top_k)
        
        
        # Stage 3: Cross-encoder final rerank
        final_results = self.rerank_with_crossencoder(query, semantic_top_indices, top_k=final_top_k)
        
        total_time = time.time() - start_total
        
        
        return final_results


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
        
        # Initialize Bible search
        self.bible_search = None
        
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
        self.semantic_topk = int(self.settings.value('semantic_topk') or 100)
        self.auto_topk = int(self.settings.value('auto_topk') or 10)

        # BM25_TOP_K: Maximum number of top results to retrieve
        self.BM25_TOP_K = int(self.settings.value('bm25_top_k') or 1000)

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
        #loading the search models 
        self.initialize_bible_search()
        
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
        self.prev_transcription = ""
        
        # Performance monitoring
        self.stats = {
            'audio_drops': 0,
            'transcription_count': 0,
            'avg_transcription_time': 0.0
        }
        
        # Emit loading status complete
        self.loadingStatus.emit()
        
       
    
    def initialize_bible_search(self):
        """Initialize Bible search in background thread"""
        try:
            print("🔄 Initializing Bible search engine...")
            self.bible_search = BibleSearch(device_type=self.device_type)
            print("✅ Bible search engine ready!")
        except Exception as e:
            print(f"❌ Failed to initialize Bible search: {e}")
            self.bible_search = None
    
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

    def perform_auto_search(self, query):
        """Perform Bible search with the transcribed query"""
        if not self.bible_search:
            print("⚠️ Bible search not initialized yet")
            return []
        
        try:
            # Perform the search using the BibleSearch class
            results = self.bible_search.search(
                query=query,
                bm25_top_k=self.BM25_TOP_K,
                semantic_top_k=self.semantic_topk,
                final_top_k=self.auto_topk
            )
           
            
            for result in results:
                score = torch.tensor(result["score"], dtype=torch.float32)
                result["score"] = torch.sigmoid(score).mul(100).item()
            
            # Sort results by score
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
            print(f"❌ Error in auto-search: {e}")
            return ["Search error occurred. Please try again."]

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

                        if self.prev_transcription is None:
                            self.prev_transcription = text
                        else:
                            # Find the overlap between the previous transcription and the current text
                            overlap_index = -1
                            for i in range(len(self.prev_transcription)):
                                if text.startswith(self.prev_transcription[i:]):
                                    overlap_index = i
                                    break

                            if overlap_index != -1:
                                # Overlap found, merge the texts
                                merged_text = self.prev_transcription[:overlap_index] + text
                                print(f"overlap made: {merged_text}")
                                query = merged_text
                            else:
                                # No overlap, treat as separate
                                print(f"no overlap: {self.prev_transcription} {text}")
                                query = text
                            self.prev_transcription = text

                        # Perform auto-search with the transcribed text
                        
                        print(f"🔍 Looking up verse for query: {query}")
                        
                        # Perform the search
                        results = self.perform_auto_search(query)
                        print('result here')
                        for result in results:
                            print(f"  📖 {result['reference']} - Score: {result['score']}")
                        
                        # Emit the search results
                        if results:
                            confidence = 0.9  # Default confidence score
                            self.autoSearchResults.emit(results, query, confidence, len(results))
                        else:
                            print("⚠️ No search results found")

                        # Clear queue if we're falling behind
                        if self.transcription_queue.qsize() > 3:
                            print(f"⚠️ Transcription backlog: {self.transcription_queue.qsize()} tasks")

                except queue.Empty:
                    continue
        except Exception as e:
            print(f"Error in transcription loop: {e}")
            
    def stop(self):
        """
        Graceful shutdown method
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