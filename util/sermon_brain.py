"""
SermonBrain — context-aware re-ranking layer that sits between
BibleStreamMatcher and the UI.

Architecture:
    Whisper chunk
        ↓
    BibleStreamMatcher  (fast, runs every chunk)
        ↓ raw candidates
    SermonBrain.process(chunk, candidates)
        ├── TopicTracker        weights candidates by current sermon topic
        ├── ArcDetector         adjusts sensitivity by sermon phase
        ├── ClusterMemory       boosts verses consistent with confirmed history
        └── OllamaReranker      async LLM final pass (non-blocking)
        ↓
    emit to UI

Usage in transcribe_loop:
    # Init (once at startup)
    from sermon_brain import SermonBrain
    self.sermon_brain = SermonBrain(ollama_model="llama3")  # or your model name

    # Replace direct emit with brain processing
    raw_results = list(self.stream_matcher.feed(text))
    for result in self.sermon_brain.process(text, raw_results):
        self.autoSearchResults.emit([result], result['text'], result['score'], self.auto_search_size)

    # On session end / new sermon
    self.sermon_brain.reset()
"""

import re
import time
import threading
import queue
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# TOPIC TAXONOMY
# Maps broad sermon topics to sets of keywords and Bible book affinities.
# When the current topic is detected, candidate verses whose books/text
# align with that topic get a score bonus.
# ============================================================================

TOPIC_MAP = {
    "salvation": {
        "keywords": {"saved", "salvation", "grace", "faith", "believe", "eternal", "life",
                     "born", "again", "repent", "forgiven", "forgiveness", "cross", "blood",
                     "justified", "redeemed", "redemption", "gospel", "perish"},
        "book_affinity": {"John", "Romans", "Ephesians", "Galatians", "Acts", "Luke",
                          "1 John", "Hebrews", "Titus"},
        "theme_words": {"sin", "death", "gift", "free", "believe", "trust", "receive"},
    },
    "prayer": {
        "keywords": {"pray", "prayer", "praying", "intercede", "petition", "ask",
                     "seek", "knock", "fasting", "fast", "kneel", "supplications"},
        "book_affinity": {"Psalms", "Matthew", "Luke", "James", "Philippians",
                          "1 Thessalonians", "Daniel", "Nehemiah"},
        "theme_words": {"answer", "hear", "listen", "request", "throne", "boldly"},
    },
    "faith": {
        "keywords": {"faith", "trust", "believe", "doubt", "hope", "confidence",
                     "assurance", "certainty", "faithful", "faithfulness"},
        "book_affinity": {"Hebrews", "Romans", "James", "Galatians", "Matthew",
                          "Mark", "Habakkuk", "2 Corinthians"},
        "theme_words": {"mountain", "move", "mustard", "seed", "walk", "sight"},
    },
    "love": {
        "keywords": {"love", "loved", "loving", "charity", "compassion", "kindness",
                     "mercy", "tender", "affection", "beloved"},
        "book_affinity": {"1 Corinthians", "John", "1 John", "Romans", "Song of Solomon",
                          "Hosea", "Ruth"},
        "theme_words": {"greatest", "commandment", "neighbor", "enemy", "unconditional"},
    },
    "purpose": {
        "keywords": {"purpose", "plan", "calling", "called", "destiny", "designed",
                     "created", "gifts", "talent", "mission", "will", "intentional"},
        "book_affinity": {"Jeremiah", "Romans", "Ephesians", "Proverbs", "Ecclesiastes",
                          "Isaiah", "Psalm"},
        "theme_words": {"know", "future", "hope", "good", "works", "masterpiece"},
    },
    "suffering": {
        "keywords": {"suffering", "suffer", "pain", "trials", "tribulation", "hardship",
                     "affliction", "trouble", "difficult", "struggle", "grief", "sorrow",
                     "weep", "mourning", "loss", "lonely", "loneliness"},
        "book_affinity": {"Job", "Psalms", "Lamentations", "2 Corinthians", "Romans",
                          "James", "1 Peter", "Isaiah"},
        "theme_words": {"comfort", "endure", "persevere", "refine", "strengthen"},
    },
    "worship": {
        "keywords": {"worship", "praise", "glorify", "honor", "exalt", "magnify",
                     "adore", "thanksgiving", "sing", "hallelujah", "blessed"},
        "book_affinity": {"Psalms", "Revelation", "Isaiah", "1 Chronicles", "Exodus",
                          "Philippians"},
        "theme_words": {"holy", "worthy", "glory", "throne", "lifted", "high"},
    },
    "leadership": {
        "keywords": {"leader", "leadership", "serve", "servant", "shepherd", "steward",
                     "overseer", "elder", "authority", "responsibility", "example"},
        "book_affinity": {"Nehemiah", "1 Timothy", "Titus", "1 Peter", "Mark",
                          "Matthew", "Joshua", "Proverbs"},
        "theme_words": {"lead", "guide", "flock", "humble", "model", "others"},
    },
    "holy_spirit": {
        "keywords": {"spirit", "holy spirit", "ghost", "anointing", "anointed", "filled",
                     "fruits", "gifts", "tongues", "power", "fire", "wind"},
        "book_affinity": {"Acts", "John", "Galatians", "1 Corinthians", "Romans",
                          "Ezekiel", "Joel"},
        "theme_words": {"baptized", "poured", "receive", "comforter", "helper", "guide"},
    },
    "identity": {
        "keywords": {"identity", "who you are", "child", "children", "adopted", "chosen",
                     "heir", "new creation", "image", "worth", "value", "belong"},
        "book_affinity": {"Ephesians", "Galatians", "Romans", "1 John", "Genesis",
                          "Psalms", "Isaiah"},
        "theme_words": {"made", "fearfully", "wonderfully", "known", "accepted", "loved"},
    },
    "money": {
        "keywords": {"money", "wealth", "riches", "tithe", "giving", "generous",
                     "offering", "stewardship", "mammon", "treasure", "financial",
                     "prosperity", "poverty", "debt"},
        "book_affinity": {"Proverbs", "Matthew", "Luke", "Malachi", "1 Timothy",
                          "Ecclesiastes", "2 Corinthians"},
        "theme_words": {"store", "heart", "serve", "master", "first", "seek"},
    },
}

# Sermon arc phases and their characteristics
ARC_PHASES = ["intro", "scripture", "exposition", "application", "close"]

# How much each phase boosts/dampens match sensitivity (multiplier on score)
ARC_SENSITIVITY = {
    "intro":       0.6,   # preacher telling a story — be skeptical
    "scripture":   1.3,   # actively reading Bible — be very receptive
    "exposition":  1.1,   # explaining scripture — still high
    "application": 0.9,   # practical application — moderate
    "close":       0.8,   # wrapping up — lower
}

# Signals that suggest each phase
ARC_SIGNALS = {
    "intro": {
        "i want to", "let me tell you", "story", "illustration", "picture this",
        "imagine", "one day", "there was", "growing up", "i remember",
        "let me share", "few years ago", "recently", "my wife", "my kids",
    },
    "scripture": {
        "verse", "scripture", "bible says", "it says", "word of god",
        "turn with me", "chapter", "reads", "written", "thus saith",
        "the lord says", "god says", "paul writes", "jesus said",
        "according to", "in the book of",
    },
    "exposition": {
        "what that means", "in other words", "the word", "greek word",
        "hebrew word", "context", "original", "actually means",
        "notice", "observe", "look at", "pay attention", "the point is",
        "what he is saying", "what god is saying",
    },
    "application": {
        "so what", "how do we", "in your life", "practically", "today",
        "you can", "we need to", "apply", "challenge", "ask yourself",
        "this week", "starting today", "action", "step",
    },
    "close": {
        "in closing", "as we close", "in conclusion", "wrap up",
        "final thought", "last thing", "before we go", "let us pray",
        "bow your heads", "one thing", "remember this", "take away",
    },
}


# ============================================================================
# 1. TOPIC TRACKER
# ============================================================================

class TopicTracker:
    """
    Tracks what topic the sermon is currently covering using a rolling
    window of recent text. Scores candidates higher when their verse
    text aligns with the active topic.
    """

    WINDOW_CHUNKS = 10      # how many recent chunks to consider
    MIN_TOPIC_SCORE = 2     # min keyword hits to declare a topic active
    TOPIC_BONUS = 0.08      # score bonus for topic-aligned verses (0-1 scale)
    TOPIC_PENALTY = 0.04    # score penalty for clearly off-topic verses

    def __init__(self):
        self.chunk_window: deque = deque(maxlen=self.WINDOW_CHUNKS)
        self.active_topics: list = []   # ordered by confidence
        self.topic_scores: dict = {}    # topic → hit count

    def ingest(self, text: str):
        """Add a new chunk to the topic window and recompute active topics."""
        self.chunk_window.append(text.lower())
        self._recompute_topics()

    def _recompute_topics(self):
        full_text = " ".join(self.chunk_window)
        words = set(re.findall(r"\b\w+\b", full_text))

        scores = {}
        for topic, data in TOPIC_MAP.items():
            hit = sum(1 for kw in data["keywords"] if kw in words)
            if hit >= self.MIN_TOPIC_SCORE:
                scores[topic] = hit

        self.topic_scores = scores
        self.active_topics = sorted(scores, key=lambda t: scores[t], reverse=True)

    def adjust_score(self, result: dict) -> float:
        """
        Return a score delta (+/-) based on how well the verse
        aligns with the current active topic.
        """
        if not self.active_topics:
            return 0.0

        verse_text = result.get("text", "").lower()
        verse_book = result.get("book", "")
        delta = 0.0

        for topic in self.active_topics[:2]:   # top 2 active topics
            data = TOPIC_MAP[topic]
            weight = self.topic_scores[topic] / 10.0   # normalise

            # Book affinity
            if verse_book in data["book_affinity"]:
                delta += self.TOPIC_BONUS * weight

            # Theme word overlap in verse text
            theme_hits = sum(1 for w in data["theme_words"] if w in verse_text)
            if theme_hits >= 2:
                delta += self.TOPIC_BONUS * 0.5 * weight

        # Penalise if verse strongly matches a *different* topic that's not active
        inactive_topics = [t for t in TOPIC_MAP if t not in self.active_topics[:3]]
        for topic in inactive_topics:
            data = TOPIC_MAP[topic]
            if verse_book in data["book_affinity"]:
                keyword_hits = sum(1 for kw in data["keywords"] if kw in verse_text)
                if keyword_hits >= 3:
                    delta -= self.TOPIC_PENALTY

        return max(-0.15, min(0.15, delta))   # clamp

    @property
    def summary(self) -> str:
        if not self.active_topics:
            return "no topic detected"
        return ", ".join(
            f"{t}({self.topic_scores[t]})" for t in self.active_topics[:3]
        )


# ============================================================================
# 2. ARC DETECTOR
# ============================================================================

class ArcDetector:
    """
    Detects which phase of the sermon is currently happening and
    returns a sensitivity multiplier that adjusts match thresholds.
    """

    WINDOW_CHUNKS = 6

    def __init__(self):
        self.chunk_window: deque = deque(maxlen=self.WINDOW_CHUNKS)
        self.current_phase: str = "intro"
        self.phase_history: list = []
        self.chunks_in_phase: int = 0

    def ingest(self, text: str):
        self.chunk_window.append(text.lower())
        prev_phase = self.current_phase
        self.current_phase = self._detect_phase()
        if self.current_phase == prev_phase:
            self.chunks_in_phase += 1
        else:
            self.chunks_in_phase = 0
            self.phase_history.append(prev_phase)

    def _detect_phase(self) -> str:
        full_text = " ".join(self.chunk_window)
        phase_scores = {}
        for phase, signals in ARC_SIGNALS.items():
            score = sum(1 for sig in signals if sig in full_text)
            if score > 0:
                phase_scores[phase] = score

        if not phase_scores:
            # No signals — assume we're still in current phase
            return self.current_phase

        return max(phase_scores, key=lambda p: phase_scores[p])

    @property
    def sensitivity(self) -> float:
        return ARC_SENSITIVITY.get(self.current_phase, 1.0)

    @property
    def summary(self) -> str:
        return f"{self.current_phase} (sensitivity={self.sensitivity:.1f})"


# ============================================================================
# 3. CLUSTER MEMORY
# ============================================================================

class ClusterMemory:
    """
    Remembers confirmed verses and uses them to boost candidates that
    are contextually consistent with the sermon's scripture history.

    Consistency means:
        - Same book as a recently confirmed verse
        - Adjacent chapter to a recently confirmed verse
        - Same broad theme (OT wisdom, NT epistles, gospels, etc.)
    """

    MEMORY_SIZE = 12        # how many confirmed verses to remember
    ADJACENCY_BONUS = 0.06  # bonus for same-book nearby chapter
    SAME_BOOK_BONUS = 0.04
    OUTLIER_PENALTY = 0.05  # penalty when no relation to any recent verse

    # Broad canonical groupings
    CANON_GROUPS = {
        "torah":        {"Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"},
        "history":      {"Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
                         "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
                         "Ezra", "Nehemiah", "Esther"},
        "wisdom":       {"Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon"},
        "major_proph":  {"Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel"},
        "minor_proph":  {"Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
                         "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
                         "Malachi"},
        "gospels":      {"Matthew", "Mark", "Luke", "John"},
        "acts":         {"Acts"},
        "paul":         {"Romans", "1 Corinthians", "2 Corinthians", "Galatians",
                         "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
                         "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus",
                         "Philemon"},
        "general_ep":   {"Hebrews", "James", "1 Peter", "2 Peter",
                         "1 John", "2 John", "3 John", "Jude"},
        "apocalyptic":  {"Revelation"},
    }

    def __init__(self):
        self.confirmed: deque = deque(maxlen=self.MEMORY_SIZE)  # list of result dicts
        self._book_counts: defaultdict = defaultdict(int)
        self._group_counts: defaultdict = defaultdict(int)

    def record(self, result: dict):
        """Call this when a verse is confirmed and emitted."""
        self.confirmed.append(result)
        book = result.get("book", "")
        self._book_counts[book] += 1
        group = self._book_to_group(book)
        if group:
            self._group_counts[group] += 1

    def adjust_score(self, result: dict) -> float:
        if not self.confirmed:
            return 0.0

        book = result.get("book", "")
        chapter = result.get("chapter", 0)
        delta = 0.0

        # Same book bonus
        if self._book_counts.get(book, 0) > 0:
            delta += self.SAME_BOOK_BONUS

        # Adjacent chapter bonus (within 3 chapters of a recent confirmed verse)
        for prev in self.confirmed:
            if prev.get("book") == book:
                chapter_dist = abs(int(prev.get("chapter", 0)) - int(chapter))
                if chapter_dist <= 3:
                    delta += self.ADJACENCY_BONUS
                    break

        # Canonical group affinity
        group = self._book_to_group(book)
        if group and self._group_counts.get(group, 0) > 0:
            delta += 0.02

        # Outlier penalty — if nothing in history relates at all
        if delta == 0.0 and len(self.confirmed) >= 4:
            delta -= self.OUTLIER_PENALTY

        return max(-0.10, min(0.12, delta))

    def _book_to_group(self, book: str) -> Optional[str]:
        for group, books in self.CANON_GROUPS.items():
            if book in books:
                return group
        return None

    @property
    def summary(self) -> str:
        if not self.confirmed:
            return "no history"
        recent = [f"{r['book']} {r['chapter']}:{r['verse']}" for r in list(self.confirmed)[-3:]]
        return "recent: " + ", ".join(recent)


# ============================================================================
# 4. OLLAMA RERANKER  (async, non-blocking)
# ============================================================================

@dataclass
class PendingBatch:
    chunk_text: str
    candidates: list           # list of result dicts
    submitted_at: float = field(default_factory=time.time)
    callback: object = None    # callable(reranked_results)


class OllamaReranker:
    """
    Async LLM re-ranker using a local Ollama model.

    - Runs in a background thread so it never blocks transcription
    - Falls back to pass-through if Ollama is unavailable or slow
    - Batches candidates every N chunks for efficiency
    """

    TIMEOUT_S = 4.0       # max seconds to wait for Ollama response
    BATCH_EVERY = 3       # process every N chunks (not every single one)
    MAX_CANDIDATES = 5    # send top N candidates to LLM

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self._queue: queue.Queue = queue.Queue()
        self._available: bool = False
        self._chunk_counter: int = 0
        self._pending_candidates: list = []
        self._pending_text: list = []

        # Test connectivity and start worker thread
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._check_availability()

    def _check_availability(self):
        """Ping Ollama in background — sets self._available."""
        def _ping():
            try:
                import urllib.request
                req = urllib.request.urlopen(
                    f"{self.host}/api/tags", timeout=2.0
                )
                if req.status == 200:
                    self._available = True
                    print(f"✅ OllamaReranker: connected to {self.host} (model: {self.model})")
            except Exception as e:
                self._available = False
                print(f"⚠️  OllamaReranker: Ollama not available ({e}) — passing through results unchanged")
        threading.Thread(target=_ping, daemon=True).start()

    def submit(self, chunk_text: str, candidates: list, callback) -> bool:
        """
        Submit candidates for async re-ranking.
        callback(results) will be called with re-ranked list when done.
        Returns True if submitted, False if passed through immediately.
        """
        if not self._available or not candidates:
            callback(candidates)
            return False

        self._chunk_counter += 1
        self._pending_text.append(chunk_text)
        self._pending_candidates.extend(candidates)

        if self._chunk_counter % self.BATCH_EVERY == 0:
            # Deduplicate pending candidates by reference
            seen = set()
            deduped = []
            for c in self._pending_candidates:
                ref = c.get("reference", "")
                if ref not in seen:
                    seen.add(ref)
                    deduped.append(c)

            batch = PendingBatch(
                chunk_text=" ".join(self._pending_text[-3:]),
                candidates=deduped[:self.MAX_CANDIDATES],
                callback=callback,
            )
            self._queue.put(batch)
            self._pending_candidates = []
            self._pending_text = []
            return True

        # Not yet at batch boundary — pass through for now
        callback(candidates)
        return False

    def _worker_loop(self):
        while True:
            try:
                batch: PendingBatch = self._queue.get(timeout=1.0)
                reranked = self._call_ollama(batch)
                if batch.callback:
                    batch.callback(reranked)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"OllamaReranker worker error: {e}")

    def _call_ollama(self, batch: PendingBatch) -> list:
        """Call Ollama and return re-ranked candidate list."""
        import json, urllib.request, urllib.error

        candidates = batch.candidates
        if not candidates:
            return candidates

        # Build prompt
        candidate_lines = "\n".join(
            f"{i+1}. [{c['reference']}] {c['text'][:120]}"
            for i, c in enumerate(candidates)
        )

        prompt = f"""You are a Bible verse identification assistant for a sermon transcription system.

The preacher just said:
\"{batch.chunk_text}\"

These Bible verses were detected as possible matches:
{candidate_lines}

Task: Return ONLY a JSON array of the candidate numbers that are genuinely being quoted or referenced, ordered by confidence. Exclude any that are false positives (coincidental word overlap, not actual scripture being quoted).

Example response: [2, 1] or [3] or []

JSON array only, no explanation:"""

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as resp:
                data = json.loads(resp.read())
                raw = data.get("response", "").strip()

                # Parse the JSON array from response
                match = re.search(r"\[[\d,\s]*\]", raw)
                if not match:
                    return candidates  # fallback: pass through

                indices = json.loads(match.group())
                if not indices:
                    return []   # LLM says none are genuine

                reranked = []
                for idx in indices:
                    if 1 <= idx <= len(candidates):
                        reranked.append(candidates[idx - 1])

                # Append any remaining candidates not mentioned (passed through at end)
                mentioned = {candidates[i-1]["reference"] for i in indices if 1 <= i <= len(candidates)}
                for c in candidates:
                    if c["reference"] not in mentioned:
                        reranked.append(c)

                print(f"  [OllamaReranker] {len(candidates)} → {len(reranked)} "
                      f"(kept: {[c['reference'] for c in reranked[:3]]})")
                return reranked

        except urllib.error.URLError:
            self._available = False
            print("⚠️  OllamaReranker: lost connection — falling back to pass-through")
            return candidates
        except Exception as e:
            print(f"  OllamaReranker error: {e} — passing through")
            return candidates


# ============================================================================
# 5. SERMON BRAIN  — main orchestrator
# ============================================================================

class SermonBrain:
    """
    Orchestrates all brain components. Drop into transcribe_loop between
    stream_matcher.feed() and autoSearchResults.emit().

    Quick start (no Ollama):
        brain = SermonBrain(use_ollama=False)

    Quick start (with Ollama):
        brain = SermonBrain(use_ollama=True, ollama_model="llama3")

    Usage:
        raw = list(self.stream_matcher.feed(text))
        for result in brain.process(text, raw):
            self.autoSearchResults.emit(...)

        # When a verse is confirmed shown in UI:
        brain.record_confirmed(result)

        # On new sermon session:
        brain.reset()
    """

    MIN_EMIT_SCORE = 0.72

    def __init__(self, use_ollama: bool = False,
                 ollama_model: str = "llama3",
                 ollama_host: str = "http://localhost:11434"):
        self.topic_tracker  = TopicTracker()
        self.arc_detector   = ArcDetector()
        self.cluster_memory = ClusterMemory()

        self.use_ollama = use_ollama
        if use_ollama:
            self.reranker = OllamaReranker(model=ollama_model, host=ollama_host)
        else:
            self.reranker = None
            print("SermonBrain: running without Ollama (topic+arc+cluster only)")

        self._pending_emit: list = []
        self._emit_lock = threading.Lock()
        self._emit_callback = None

    def set_emit_callback(self, callback):
        """
        Optional: set a callback for async Ollama results.
        callback(results: list) will be called from the Ollama worker thread.
        In Qt, make sure to emit via a signal from this callback — don't
        update UI directly from a non-Qt thread.

        If not set, async results are buffered in self._pending_emit and
        can be retrieved via drain_pending().
        """
        self._emit_callback = callback

    def process(self, chunk_text: str, raw_results: list) -> list:
        """
        Main entry point. Takes raw stream matcher results and returns
        brain-adjusted results ready to emit.

        Synchronous path:
            - Topic, arc, cluster adjustments applied immediately
            - Results below MIN_EMIT_SCORE filtered out
            - Remaining results returned synchronously

        Async path (Ollama):
            - Top candidates sent to Ollama in background
            - If emit_callback is set, confirmed results arrive via callback
            - Otherwise drain_pending() returns buffered results
        """
        # Update context trackers
        self.topic_tracker.ingest(chunk_text)
        self.arc_detector.ingest(chunk_text)

        if not raw_results:
            return []

        arc_sensitivity = self.arc_detector.sensitivity

        adjusted = []
        for result in raw_results:
            r = dict(result)   # don't mutate original

            base_score = float(r.get("score", 0.0))

            # Apply brain adjustments
            topic_delta   = self.topic_tracker.adjust_score(r)
            cluster_delta = self.cluster_memory.adjust_score(r)

            # Arc sensitivity scales the deltas (intro phase = smaller nudges)
            adjusted_score = base_score + (topic_delta + cluster_delta) * arc_sensitivity

            # During scripture phase, lower the effective threshold slightly
            effective_min = self.MIN_EMIT_SCORE
            if self.arc_detector.current_phase == "scripture":
                effective_min -= 0.04
            elif self.arc_detector.current_phase == "intro":
                effective_min += 0.04

            if adjusted_score < effective_min:
                print(
                    f"  [Brain] DROP {r['reference']} "
                    f"score {base_score:.2f}→{adjusted_score:.2f} "
                    f"(topic={topic_delta:+.2f} cluster={cluster_delta:+.2f} "
                    f"arc={arc_sensitivity:.1f})"
                )
                continue

            r["score"] = round(adjusted_score, 4)
            r["_brain_topic"]   = self.topic_tracker.summary
            r["_brain_arc"]     = self.arc_detector.summary
            r["_brain_cluster"] = self.cluster_memory.summary

            adjusted.append(r)

        if not adjusted:
            return []

        # Record sync results in cluster memory
        for r in adjusted:
            if not r.get("is_continuation"):
                self.cluster_memory.record(r)

        # Submit to Ollama async reranker (only if enabled)
        if self.use_ollama and self.reranker:
            def _ollama_callback(reranked: list):
                if not reranked:
                    return
                for r in reranked:
                    if not r.get("is_continuation"):
                        self.cluster_memory.record(r)
                if self._emit_callback:
                    self._emit_callback(reranked)
                else:
                    with self._emit_lock:
                        self._pending_emit.extend(reranked)

            self.reranker.submit(chunk_text, adjusted, _ollama_callback)

        print(
            f"  [Brain] {len(raw_results)} raw → {len(adjusted)} adjusted | "
            f"topic: {self.topic_tracker.summary} | "
            f"arc: {self.arc_detector.summary}"
        )

        return adjusted

    def record_confirmed(self, result: dict):
        """
        Call this when a verse is confirmed shown in the UI (e.g. user
        accepts it or it auto-displays). Updates cluster memory.
        """
        self.cluster_memory.record(result)

    def drain_pending(self) -> list:
        """
        Returns any async Ollama results buffered since last drain.
        Call this periodically if you're not using set_emit_callback().
        """
        with self._emit_lock:
            results = list(self._pending_emit)
            self._pending_emit.clear()
        return results

    def reset(self):
        """Full reset for a new sermon session."""
        self.topic_tracker  = TopicTracker()
        self.arc_detector   = ArcDetector()
        self.cluster_memory = ClusterMemory()
        self._pending_emit  = []
        print("SermonBrain reset for new session")

    @property
    def status(self) -> dict:
        if self.use_ollama and self.reranker:
            ollama_status = "available" if self.reranker._available else "unavailable"
        else:
            ollama_status = "disabled"
        return {
            "topic":   self.topic_tracker.summary,
            "arc":     self.arc_detector.summary,
            "cluster": self.cluster_memory.summary,
            "ollama":  ollama_status,
        }