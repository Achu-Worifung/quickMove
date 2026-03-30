import os
import re
import threading
import unicodedata
from typing import Type

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
from nltk.tokenize import word_tokenize
from PyQt5.QtCore import QSettings
from rapidfuzz import fuzz
from widgets.SearchWidget import QThread, pyqtSignal

from util.classifier import Classifier
from util.biblesearch import BibleSearch
from util.extract_verse import extract_bible_reference
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
        self.stop_words     = set(stopwords.words('english'))

        # Sliding window state (used only for short turns < MIN_TURN_WORDS)
        self.word_buffer    = []
        self.WINDOW_SIZE    = 12
        self.STRIDE         = 6
        self.MIN_TURN_WORDS = 4

        # Verse buffer: keeps recent high-confidence results for continuation matching
        self.verse_buffer = {}

        self._pause     = False
        self._stop      = False
        self.state_lock = threading.Lock()

        self.settings   = QSettings("MyApp", "AutomataSimulator")
        self.stop_words = set(stopwords.words('english'))
        self.api_key    = os.getenv("ASSEMBLYAI_API_KEY")

        # Settings
        self.auto_search_size = int(self.settings.value('auto_length') or 4)
        self.speech_model     = self.settings.value('assemblyai_model') or 'u3-rt-pro'

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        with self.state_lock:
            self._stop = True
        print('Stopping transcription thread...')

    def pause_transcription(self):
        """Toggle pause. Returns new pause state (True = paused)."""
        with self.state_lock:
            self._pause = not self._pause
            new_value = self._pause

        if new_value:
            self.statusUpdate.emit("paused")
            print('Transcription paused')
        else:
            self.statusUpdate.emit("listening")
            print('Transcription resumed')

        return new_value

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
    
    def _extract_bible_segment(self, text: str) -> str:
        """
        Generate all sliding windows across the text, classify them in a
        single batched call, then return the text from the first Bible
        window onward.
        """
        words  = text.split()
        WINDOW = 6
        STEP   = 2

        if len(words) <= WINDOW:
            return text  # too short to slice — use as-is

        # Build all windows at once
        windows = []
        indices = []
        for i in range(0, max(1, len(words) - WINDOW + 1), STEP):
            window_text = re.sub(r'[^\w\s]', '', ' '.join(words[i:i + WINDOW]).lower()).strip()
            if window_text:
                windows.append(window_text)
                indices.append(i)

        if not windows:
            return text

        # Single batched inference call — all windows at once
        results = self.classifier.classify(windows)

        # Find the first window classified as Bible
        for result, start_idx in zip(results, indices):
            label, confidence = result
            if label != "non bible":
                segment = ' '.join(words[start_idx:]).strip()
                print(f"Bible segment starts at word {start_idx}: '{segment}'")
                return segment

        return text 

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def initialize_bible_search(self):
        try:
            print("Initialising Bible search engine...")
            self.bible_search = BibleSearch()
            print("Bible search engine ready.")
        except Exception as e:
            print(f"Failed to initialise Bible search: {e}")
            self.bible_search = None

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()

        # Remove "Original: ... Paraphrase: " blocks
        text = re.sub(r'original:.*?paraphrase:\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'^paraphrase:\s*', '', text, flags=re.IGNORECASE)

        # Clean escape sequences and quotes
        text = text.replace('\\"', '"')
        text = text.strip('"').strip("'").strip()
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\\', '')

        # Remove punctuation (preserve colons for verse references)
        text = re.sub(r'[^\w\s:]', '', text)

        # Collapse whitespace
        text = ' '.join(text.split())

        # Normalize unicode to ASCII
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        return text

    # ------------------------------------------------------------------
    # AssemblyAI streaming loop
    # ------------------------------------------------------------------

    def transcribe_loop(self):

        def on_begin(client: Type[StreamingClient], event: BeginEvent):
            print(f"AssemblyAI session started: {event.id}")
            self.statusUpdate.emit("listening")

        def on_turn(client: Type[StreamingClient], event: TurnEvent):
            # Partial turns: stream live text to UI only
            if not event.end_of_turn:
                if event.transcript:
                    self.guitextReady.emit(event.transcript)
                    text = event.transcript.strip()
                    if text:
                        self._process_turn(text) #process partial turns too, for more responsive searching (can be noisy, but UI can handle it)
                return

            # Final turn
            if self._pause or self._stop:
                return

            text = event.transcript.strip()
            if not text:
                return

            self.guitextReady.emit(text)
            self.statusUpdate.emit("processing")

            try:
                self._process_turn(text)
            except Exception as e:
                import traceback
                print(f"Error processing turn: {e}")
                traceback.print_exc()
            finally:
                self.statusUpdate.emit("listening")

        def on_terminated(client: Type[StreamingClient], event: TerminationEvent):
            print(f"Session ended — {event.audio_duration_seconds:.1f}s processed")

        def on_error(client: Type[StreamingClient], error: StreamingError):
            print(f"AssemblyAI error: {error}")
            self.statusUpdate.emit("error")

        client = StreamingClient(
            StreamingClientOptions(
                api_key=self.api_key,
                api_host="streaming.assemblyai.com",
            )
        )
        client.on(StreamingEvents.Begin,       on_begin)
        client.on(StreamingEvents.Turn,        on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error,       on_error)

        client.connect(StreamingParameters(
            sample_rate=16000,
            speech_model=self.speech_model,
            max_turn_silence=900,
            end_of_turn_confidence_threshold=0.5,
        ))

        # Create the microphone stream
        mic_stream = aai.extras.MicrophoneStream(sample_rate=16000)

        def stream_generator():
            """Yields audio chunks only as long as self._stop is False."""
            for chunk in mic_stream:
                if self._stop:
                    break
                # Respect the pause flag here to save processing/API costs
                if not self._pause:
                    yield chunk

        try:
            # Pass our generator instead of the raw mic_stream
            client.stream(stream_generator())
        finally:
            print("Cleaning up AssemblyAI client...")
            # This ensures the 'Termination' event is actually sent to the server
            client.disconnect(terminate=True)
            mic_stream.close()

    # ------------------------------------------------------------------
    # Turn processing pipeline
    # ------------------------------------------------------------------

    def _process_turn(self, text: str):
        """
        Full pipeline for a finalised turn:
          1. Spoken reference detection  → emit directly, skip search
          2. Decide query chunk          → full turn or sliding window
          3. Continuation match          → re-emit buffered verse, skip search
          4. Classification              → skip if non-Bible and no keywords
          5. Search + deduplicate        → emit results
        """

        # 1. Explicit spoken reference (e.g. "John 3:16")
        spoken_references = extract_bible_reference(text)
        if spoken_references:
            reference_dict = [
                {
                    'reference': ref['full'],
                    'book':      ref.get('book', ''),
                    'chapter':   ref.get('chapter', ''),
                    'verse':     ref.get('verse', ''),
                    'text':      '',
                    'score':     1.0,
                }
                for ref in spoken_references
            ]
            print(f"Spoken reference(s) found: {[r['reference'] for r in reference_dict]}")
            self.autoSearchResults.emit(reference_dict, '', 0.0, self.auto_search_size)
            return

        # 2. Decide query chunk
        #    Long turns are used directly — AssemblyAI's utterance boundaries are clean.
        #    Short turns accumulate in the sliding window to avoid searching on fragments.
        word_count = len(text.split())

        if word_count >= self.MIN_TURN_WORDS:
            chunk = text
            self.word_buffer.clear()  # flush stale words so they don't bleed into future windows
        else:
            chunk = self._get_sliding_window_chunk(text)
            if chunk is None:
                return  # not enough words accumulated yet

        # 3. Continuation match against verse buffer
        if self.verse_buffer and self._check_continuation(text, chunk):
            return

        # 4. Classify
        if not self._is_bible_content(chunk):
            print("Non-Bible content — clearing verse buffer, skipping search")
            self.verse_buffer = {}
            return

        # 5. Search
        self._search_and_emit(chunk)

    def _check_continuation(self, text: str, chunk: str) -> bool:
        """
        Check if `text` is a continuation of a verse already in the buffer.
        Returns True and emits if a match is found above threshold.
        """
        best_score      = 0
        best_match_info = None

        for verse_data in self.verse_buffer.values():
            v_text    = verse_data.get('text', '') if isinstance(verse_data, dict) else str(verse_data)
            alignment = fuzz.partial_ratio_alignment(text.lower(), v_text.lower())
            if alignment.score > best_score:
                best_score      = alignment.score
                best_match_info = {
                    'data': verse_data,
                    'text': v_text,
                    'end':  alignment.dest_end,
                }

        if best_score > 70 and best_match_info:
            d             = best_match_info['data']
            target_text   = best_match_info['text']
            matched_chunk = target_text[:best_match_info['end']]
            threshold     = self._confidence_threshold()

            print(f"Continuation match: {d.get('book')} {d.get('chapter')}:{d.get('verse')} ({best_score:.0f}%)")
            self.autoSearchResults.emit(
                [{
                    'reference':       d.get('reference', 'Unknown'),
                    'book':            d.get('book', ''),
                    'chapter':         d.get('chapter', ''),
                    'verse':           d.get('verse', ''),
                    'text':            target_text,
                    'score':           '100%',
                    'is_continuation': True,
                }],
                matched_chunk,
                threshold,
                self.auto_search_size,
            )
            return True

        print(f"No continuation match (best: {best_score:.0f}%) — proceeding to classify")
        return False

    def _is_bible_content(self, chunk: str) -> bool:
        """Classify chunk. Returns True if Bible-related content."""
        normalized = self.normalize_text(chunk)
       
        print('Classifying chunk:', normalized)
        if not normalized.strip():
            print("Chunk is empty after removing stopwords — treating as non-Bible")
            return False
        label, confidence = self.classifier.classify([normalized])[0]
        print(f"Classification: {label} ({confidence:.2%})")

        if label != "non bible":
            return True

        # Keyword fallback on original text (pre-normalization)
        keyword_pattern = r"\b(god|lord|jesus|christ|heaven|hell)\b"
        return bool(re.search(keyword_pattern, chunk, re.IGNORECASE))

    def _search_and_emit(self, query: str):
        """Search, filter, deduplicate, update verse buffer, and emit results."""
        threshold      = self._confidence_threshold()
        search_results = self.perform_auto_search(query)

        filtered = []
        for r in search_results:
            try:
                score_val = float(str(r.get('score', 0)).replace('%', ''))
            except ValueError:
                score_val = 0.0

            if score_val >= threshold:
                ref_key = f"{r.get('book', '')} {r.get('chapter', '')}:{r.get('verse', '')}"
                self.verse_buffer[ref_key] = {
                    'text':      r['text'],
                    'score':     score_val,
                    'version':   r.get('version', ''),
                    'book':      r.get('book', ''),
                    'chapter':   r.get('chapter', ''),
                    'verse':     r.get('verse', ''),
                    'reference': r.get('reference', ref_key),
                }
                filtered.append(r)

        # Deduplicate: keep highest-scoring entry per verse
        verse_map = {}
        for r in filtered:
            verse_key = f"{r['book'].lower()} {r['chapter']}:{r['verse']}"
            score     = float(str(r.get('score', 0)).replace('%', ''))
            existing  = float(str(verse_map.get(verse_key, {}).get('score', 0)).replace('%', ''))
            if verse_key not in verse_map or score > existing:
                verse_map[verse_key] = r

        deduped = list(verse_map.values())
        if deduped:
            print(f"Emitting {len(deduped)} result(s) for '{query}'")
            self.autoSearchResults.emit(deduped, query, threshold, self.auto_search_size)
        else:
            print(f"No results above threshold ({threshold:.0%}) for '{query}'")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sliding_window_chunk(self, text: str):
        """
        Accumulate words from short turns and return a fixed-size chunk
        once WINDOW_SIZE words are available, then stride forward.
        """
        self.word_buffer.extend(text.split())

        if len(self.word_buffer) >= self.WINDOW_SIZE:
            chunk            = " ".join(self.word_buffer[:self.WINDOW_SIZE])
            self.word_buffer = self.word_buffer[self.STRIDE:]
            return chunk

        return None

    def _confidence_threshold(self) -> float:
        return float(self.settings.value("bible_confidence") or 60) / 100

    def perform_auto_search(self, query: str) -> list:
        if not self.bible_search:
            print("Bible search not initialised yet")
            return []
        try:
            return self.bible_search.search(self.normalize_text(query))
        except Exception as e:
            print(f"Error in auto-search: {e}")
            return []