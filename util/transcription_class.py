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
        self.no_match_count = 0
        
        # State Management
        self._pause     = False
        self._stop      = False
        self.state_lock = threading.Lock()
        self.verse_lock = threading.Lock()

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
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.WINDOW_SIZE = 8      # Words per chunk
        self.STRIDE = 4           # Overlap: WINDOW_SIZE - STRIDE = 4 words overlap
        self.MIN_CHUNK_WORDS = 4  # Minimum words to classify

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
        try:
            self.classifier = Classifier()
            self.classifier.load_classifier()
        except Exception as e:
            self.statusUpdate.emit(f"classifier error: {e}")
            self.classifier = None
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

    def _parse_score(self, score) -> float:
        if isinstance(score, (int, float)):
            return score if score <= 1 else score / 100
        return float(str(score).replace('%', '')) / 100
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
            max_turn_silence=200,
            end_of_turn_confidence_threshold=0.5,
        ))

        try:
            mic_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        except Exception as e:
            self.statusUpdate.emit(f"mic error: {e}")
            return

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
    def _create_overlapping_chunks(self, text: str) -> List[Dict]:
        """Create overlapping word chunks from text for batch classification."""
        words = text.split()
        
        # Configurable chunk size and stride
        WINDOW_SIZE = getattr(self, 'WINDOW_SIZE', 8)
        STRIDE = getattr(self, 'STRIDE', 4)
        MIN_CHUNK_WORDS = 4
        
        if len(words) < MIN_CHUNK_WORDS:
            return []
        
        # If text is shorter than window, return single chunk
        if len(words) <= WINDOW_SIZE:
            return [{'text': text, 'start': 0, 'end': len(words)}]
        
        chunks = []
        for i in range(0, len(words) - WINDOW_SIZE + 1, STRIDE):
            chunk_words = words[i:i + WINDOW_SIZE]
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'start': i,
                'end': i + WINDOW_SIZE
            })
        
        # Add final chunk if we haven't covered the end
        if chunks and chunks[-1]['end'] < len(words):
            remaining = words[chunks[-1]['start'] + STRIDE:]
            if len(remaining) >= MIN_CHUNK_WORDS:
                chunks.append({
                    'text': ' '.join(remaining),
                    'start': chunks[-1]['start'] + STRIDE,
                    'end': len(words)
                })
        
        return chunks

    def _process_turn(self, text: str):
        """Process transcription with a persistent buffer for short turns."""
        incoming_words = text.split()
        
        # 1. Check for explicit references (Fast Path)
        refs = extract_bible_reference(text)
        if refs:
            # If we found a reference, we likely don't need the previous buffer
            self.word_buffer = [] 
            res = [{**r, 'score': 1.0, 'ref': r['full']} for r in refs]
            self.autoSearchResults.emit(res, text, 0.0, self.auto_search_size)
            return

        # 2. Combine with previous short fragments
        combined_words = self.word_buffer + incoming_words
        combined_text = ' '.join(combined_words)

        # 3. Create chunks
        chunks = self._create_overlapping_chunks(combined_text)
        
        # Check if we still don't have enough words to form a valid chunk
        if not chunks:
            print(f"[Buffer] Text too short ({len(combined_words)} words). Saving to buffer.")
            self.word_buffer = combined_words
            
            # Optional: Limit buffer size so it doesn't grow forever if no matches found
            if len(self.word_buffer) > 30:
                self.word_buffer = self.word_buffer[-20:]
            return

        # 4. We have enough to process, so clear the buffer
        print(f"[Chunk] Processing {len(combined_words)} words (Buffer included)")
        self.word_buffer = [] 

        # 5. Batch classify all chunks
        chunk_texts = [c['text'] for c in chunks]
        try:
            results = self.classifier.classify(chunk_texts) if self.classifier else []
            print(f"[Classifier] Results: {results}")
        except Exception as e:
            print(f"[Classifier] Error: {e}")
            results = [("non bible", 0.0)] * len(chunks)

        # 4. Attach scores to chunks
        for i, (label, conf) in enumerate(results):
            chunks[i]['label'] = label
            chunks[i]['confidence'] = conf
            print(f"[Chunk {i}] '{chunks[i]['text'][:40]}...' -> {label} ({conf:.2f})")

        # 5. Filter Bible chunks and pick best
        bible_chunks = [c for c in chunks if c['label'] != "non bible" and c['confidence'] > 0.65]
        
        if not bible_chunks:
            self.no_match_count += 1
            if self.no_match_count > 3:
                with self.verse_lock:
                    self.verse_buffer = {}
            print(f"[Chunk] No Bible chunks found (count: {self.no_match_count})")
            return
        else:
            self.no_match_count = 0

        # 6. Select highest confidence chunk for search
        best_chunk = max(bible_chunks, key=lambda x: x['confidence'])
        print(f"[Chunk] Selected: '{best_chunk['text'][:50]}...' ({best_chunk['confidence']:.2f})")

        # 7. Search and emit
        threshold = self._confidence_threshold()
        results = self.perform_auto_search(best_chunk['text'])
        print('auto search results:', results)

        verse_map = {}
        with self.verse_lock:
            for r in results:
                score = self._parse_score(r.get('score', 0))
                if score >= threshold:
                    key = f"{r['book'].lower()} {r['chapter']}:{r['verse']}"
                    if key not in verse_map or score > self._parse_score(verse_map[key].get('score', 0)):
                        verse_map[key] = r
                        self.verse_buffer[key] = r

            print('verse_map:', verse_map)
            deduped = sorted(verse_map.values(), key=lambda x: self._parse_score(x.get('score', 0)), reverse=True)
            if deduped:
                print(f"[Search] Found {len(deduped)} verses, emitting...")
                self.autoSearchResults.emit(deduped, best_chunk['text'], threshold, self.auto_search_size)
            else:
                print(f"[Search] No verses above threshold ({threshold:.2f})")

    def _search_and_emit(self, query: str):
        search_query = self._extract_bible_segment(query)
        if not search_query:
            self.verse_buffer = {} # Conversation shifted, clear history
            return

        threshold = self._confidence_threshold()
        results = self.perform_auto_search(search_query)

        verse_map = {}
        for r in results:
            score = self._parse_score(r.get('score', 0))
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