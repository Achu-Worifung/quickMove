import os
import re
import threading
import unicodedata
import math
import time
from typing import Type, Optional, List, Dict

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
import nltk
from nltk.corpus import stopwords
from PyQt5.QtCore import QSettings, QThread, pyqtSignal, pyqtSlot
from rapidfuzz import fuzz

from util.classifier import Classifier
from util.biblesearch import BibleSearch
from util.extract_verse import extract_bible_reference

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TranscriptionWorker(QThread):
    finished          = pyqtSignal()
    autoSearchResults = pyqtSignal(list, str, float, int)
    guitextReady      = pyqtSignal(str)
    loadingStatus     = pyqtSignal()
    statusUpdate      = pyqtSignal(str)

    def __init__(self, parent=None, search_page=None):
        super().__init__(parent)
        self.record_page  = parent
        self.search_page  = search_page
        self.classifier   = None
        self.bible_search = None
        
        # State Management
        self._pause     = False
        self._stop      = False
        self.state_lock = threading.Lock()

        # Sliding window state for short turns
        self.word_buffer    = []
        self.WINDOW_SIZE    = 12
        self.STRIDE         = 6
        self.MIN_TURN_WORDS = 4

        # Verse buffer for continuity matching
        self.verse_buffer = {}

        self.settings   = QSettings("MyApp", "AutomataSimulator")
        self.stop_words = set(stopwords.words('english'))
        self.api_key    = os.getenv("ASSEMBLYAI_API_KEY")

        # Config
        self.auto_search_size = int(self.settings.value('auto_length') or 4)
        self.speech_model     = self.settings.value('assemblyai_model') or 'u3-rt-pro'

    # ------------------------------------------------------------------
    # Lifecycle & Control
    # ------------------------------------------------------------------

    def stop(self):
        """Signals the loop to break and the client to disconnect."""
        with self.state_lock:
            self._stop = True
        print('Stopping transcription worker...')

    def pause_transcription(self):
        with self.state_lock:
            self._pause = not self._pause
            paused = self._pause
        
        self.statusUpdate.emit("paused" if paused else "listening")
        return paused

    def run(self):
        """Main QThread entry point."""
        print("Starting transcription worker...")
        self.classifier = Classifier()
        self.classifier.load_classifier()
        self.initialize_bible_search()

        self.loadingStatus.emit()
        self.statusUpdate.emit("listening")

        try:
            self.transcribe_loop()
        except Exception as e:
            import traceback
            print(f"Fatal error in transcription worker: {e}")
            traceback.print_exc()
        finally:
            self.classifier.offload_classifier()
            self.finished.emit()

    # ------------------------------------------------------------------
    # AssemblyAI Streaming Logic
    # ------------------------------------------------------------------

    def transcribe_loop(self):
        def on_begin(client, event: BeginEvent):
            self.statusUpdate.emit("listening")

        def on_turn(client, event: TurnEvent):
            if not event.end_of_turn:
                if event.transcript:
                    self.guitextReady.emit(event.transcript)
                return

            # Finalized turn logic
            if self._stop: return
            
            text = event.transcript.strip()
            if not text: return

            self.guitextReady.emit(text)
            # self.statusUpdate.emit("processing")
            try:
                self._process_turn(text)
            finally:
                if not self._stop:
                    self.statusUpdate.emit("listening")

        def on_terminated(client, event: TerminationEvent):
            print(f"Session terminated: {event.audio_duration_seconds:.1f}s")

        def on_error(client, error: StreamingError):
            print(f"AssemblyAI Error: {error}")
            self.statusUpdate.emit("error")

        client = StreamingClient(StreamingClientOptions(api_key=self.api_key))
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)

        client.connect(StreamingParameters(
            sample_rate=16000,
            speech_model=self.speech_model,
            max_turn_silence=900,
            end_of_turn_confidence_threshold=0.5,
        ))

        mic_stream = aai.extras.MicrophoneStream(sample_rate=16000)

        def stream_generator():
            for chunk in mic_stream:
                if self._stop: break
                if not self._pause:
                    yield chunk

        try:
            client.stream(stream_generator())
        finally:
            client.disconnect(terminate=True)
            mic_stream.close()

    # ------------------------------------------------------------------
    # Noise Filtering & Segmentation
    # ------------------------------------------------------------------

    def _extract_bible_segment(self, text: str) -> Optional[str]:
        """Finds the best anchor point for Bible content using batched classification."""
        words = text.split()
        WINDOW = 4
        STEP = 2

        if len(words) < WINDOW:
            label, conf = self.classifier.classify([text])[0]
            return text if (label != "non bible" and conf > 0.75) else None

        windows, indices = [], []
        for i in range(0, max(1, len(words) - WINDOW + 1), STEP):
            win_text = self.normalize_text(' '.join(words[i:i + WINDOW]))
            if win_text:
                windows.append(win_text)
                indices.append(i)

        if not windows: return None

        results = self.classifier.classify(windows)
        candidates = [(conf, idx) for (lbl, conf), idx in zip(results, indices) 
                      if lbl != "non bible" and conf > 0.82]

        if not candidates: return None

        # Pick highest confidence anchor and return tail
        _, best_idx = max(candidates, key=lambda x: x[0])
        # Include one word of leading context
        start_idx = max(0, best_idx - 1)
        return ' '.join(words[start_idx:]).strip()

    # ------------------------------------------------------------------
    # Pipeline Processing
    # ------------------------------------------------------------------

    def _process_turn(self, text: str):
        # 1. Spoken References
        refs = extract_bible_reference(text)
        if refs:
            res = [{'reference': r['full'], 'score': 1.0} for r in refs]
            self.autoSearchResults.emit(res, '', 0.0, self.auto_search_size)
            return

        # 2. Chunking
        words = text.split()
        if len(words) >= self.MIN_TURN_WORDS:
            chunk = text
            self.word_buffer.clear()
        else:
            chunk = self._get_sliding_window_chunk(text)
            if not chunk: return

        # 3. Continuation
        if self.verse_buffer and self._check_continuation(text, chunk):
            return

        # 4 & 5. Segment & Search
        self._search_and_emit(chunk)

    def _search_and_emit(self, query: str):
        search_query = self._extract_bible_segment(query)
        if not search_query:
            self.verse_buffer = {} # Conversation shifted, clear history
            return

        threshold = self._confidence_threshold()
        results = self.perform_auto_search(search_query)

        verse_map = {}
        for r in results:
            score = float(str(r.get('score', 0)).replace('%', ''))
            if score >= threshold:
                key = f"{r['book'].lower()} {r['chapter']}:{r['verse']}"
                if key not in verse_map or score > float(str(verse_map[key].get('score', 0)).replace('%', '')):
                    verse_map[key] = r
                    self.verse_buffer[key] = r # Update buffer for next turn

        deduped = sorted(verse_map.values(), key=lambda x: float(str(x.get('score', 0)).replace('%', '')), reverse=True)
        if deduped:
            self.autoSearchResults.emit(deduped, search_query, threshold, self.auto_search_size)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _get_sliding_window_chunk(self, text: str):
        self.word_buffer.extend(text.split())
        if len(self.word_buffer) >= self.WINDOW_SIZE:
            chunk = " ".join(self.word_buffer[:self.WINDOW_SIZE])
            self.word_buffer = self.word_buffer[self.STRIDE:]
            return chunk
        return None

    def _check_continuation(self, text: str, chunk: str) -> bool:
        best_score = 0
        best_match = None
        for v in self.verse_buffer.values():
            v_text = v.get('text', '')
            align = fuzz.partial_ratio_alignment(text.lower(), v_text.lower())
            if align.score > best_score:
                best_score = align.score
                best_match = (v, align.dest_end)

        if best_score > 75 and best_match:
            v, end = best_match
            self.autoSearchResults.emit([v], v_text[:end], self._confidence_threshold(), self.auto_search_size)
            return True
        return False

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'original:.*?paraphrase:\s*', '', text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r'[^\w\s]', '', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return ' '.join(text.split())

    def _confidence_threshold(self) -> float:
        return float(self.settings.value("bible_confidence") or 60) / 100

    def initialize_bible_search(self):
        try:
            self.bible_search = BibleSearch()
        except:
            self.bible_search = None

    def perform_auto_search(self, query: str) -> list:
        if not self.bible_search: return []
        try:
            return self.bible_search.search(self.normalize_text(query))
        except:
            return []