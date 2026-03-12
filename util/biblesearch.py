"""
BibleSearch — drop-in replacement using the streaming n-gram + phonetic pipeline.

Replaces the BM25 → semantic rerank → cross-encoder ONNX stack with a direct
inverted index + fuzzy alignment approach that is:
  - Faster (no model inference at query time)
  - More robust to Whisper mishearings (phonetic Soundex indexing)
  - Works on the full 5-version corpus (~155k verses)

Public API is identical to the original BibleSearch so nothing else needs to change:
    search(query, ...)  →  list of result dicts

The BibleStreamMatcher (for real-time transcription) is also exposed via:
    get_stream_matcher()  →  BibleStreamMatcher instance (built on same index)
"""

from util.util import resource_path
import os
import json
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Optional fast deps — both already in your venv
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    import difflib

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False


# ============================================================================
# 1.  TEXT UTILITIES
# ============================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "they",
    "them", "their", "he", "him", "his", "she", "her", "it", "its",
    "we", "us", "our", "you", "your", "i", "my", "me", "who", "that",
    "this", "these", "those", "not", "no", "so", "as", "if", "by",
    "from", "up", "out", "then", "than", "now", "all", "one", "two",
    "said", "say", "says", "again", "also", "just", "when", "what",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phonetic_key(word: str) -> str:
    if HAS_JELLYFISH:
        return jellyfish.soundex(word)
    w = word.upper()
    if not w:
        return ""
    return w[0] + re.sub(r"[AEIOU]", "", w[1:])


def words_to_phonetic(words: list) -> list:
    return [phonetic_key(w) for w in words if w]


def partial_ratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return _rfuzz.partial_ratio(a, b)
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    for i in range(len(b) - len(a) + 1):
        r = difflib.SequenceMatcher(None, a, b[i:i + len(a)]).ratio() * 100
        if r > best:
            best = r
    return best


def make_ngrams(seq: list, n: int = 3) -> set:
    return set(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


# ============================================================================
# 2.  VERSE ENTRY & INDEX
# ============================================================================

@dataclass
class VerseEntry:
    reference: str
    book: str
    chapter: int
    verse: int
    version: str
    raw_text: str
    norm_text: str
    norm_words: list
    phonetic_words: list
    ngrams: set
    phonetic_ngrams: set


def build_verse_entry(verse_dict: dict) -> VerseEntry:
    raw = verse_dict.get("text", "")
    norm = normalize(raw)
    words = norm.split()
    phones = words_to_phonetic(words)
    return VerseEntry(
        reference=verse_dict["reference"],
        book=verse_dict.get("book", ""),
        chapter=int(verse_dict.get("chapter", 0)),
        verse=int(verse_dict.get("verse", 0)),
        version=verse_dict.get("version", ""),
        raw_text=raw,
        norm_text=norm,
        norm_words=words,
        phonetic_words=phones,
        ngrams=make_ngrams(words),
        phonetic_ngrams=make_ngrams(phones),
    )


class VerseIndex:
    def __init__(self):
        self.text_index: dict = defaultdict(list)
        self.phone_index: dict = defaultdict(list)
        self.all_verses: list = []

    def add(self, entry: VerseEntry):
        self.all_verses.append(entry)
        for ng in entry.ngrams:
            self.text_index[ng].append(entry)
        for ng in entry.phonetic_ngrams:
            self.phone_index[ng].append(entry)

    def recall(self, query_words: list, top_k: int = 20) -> list:
        query_phones = words_to_phonetic(query_words)
        text_ngs = make_ngrams(query_words)
        phone_ngs = make_ngrams(query_phones)
        scores: dict = {}

        for ng in text_ngs:
            for entry in self.text_index.get(ng, []):
                if entry.reference not in scores:
                    scores[entry.reference] = [entry, 0]
                scores[entry.reference][1] += 2

        for ng in phone_ngs:
            for entry in self.phone_index.get(ng, []):
                if entry.reference not in scores:
                    scores[entry.reference] = [entry, 0]
                scores[entry.reference][1] += 1

        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [(e, s) for e, s in ranked[:top_k]]


# ============================================================================
# 3.  STREAM MATCHER  (real-time transcription)
# ============================================================================

@dataclass
class Candidate:
    entry: VerseEntry
    recall_score: int
    confirm_score: float = 0.0
    age: int = 0
    confirmed: bool = False


class BibleStreamMatcher:
    """
    Real-time verse matcher for the transcription loop.
    Feed Whisper chunks via feed() — yields result dicts on match/continuation.
    Call flush_candidates() when the classifier detects non-Bible content.
    """

    WINDOW_SIZE = 12
    STRIDE = 4
    MIN_RECALL_HITS = 3
    CONFIRM_THRESHOLD = 82.0
    PROMOTE_THRESHOLD = 90.0
    MIN_CONFIRM_OVERLAP = 0.30   # score is primary gate; overlap prevents stopword-only matches
    CONTINUATION_THRESHOLD = 80.0
    CONTINUATION_OVERLAP = 0.35
    MAX_CANDIDATE_AGE = 6
    MAX_CANDIDATES = 10
    EMIT_BEST_VERSION_ONLY = True
    MIN_MATCH_WORDS = 8

    def __init__(self, verse_index: VerseIndex):
        self.index = verse_index
        self.reset()

    def reset(self):
        """Full session reset — clears everything including emitted."""
        self.word_buffer: list = []
        self.candidates: list = []
        self.emitted: set = set()
        self.chunk_count: int = 0

    def flush_candidates(self):
        """
        Clear candidates and word buffer but KEEP the emitted set.
        This prevents the same false-positive from re-firing on the next
        similar chunk after a flush. emitted is only cleared on reset().
        """
        self.candidates = []
        self.word_buffer = []

    def feed(self, raw_text: str):
        norm = normalize(raw_text)
        new_words = norm.split()
        if not new_words:
            return

        new_words = self._dedup_overlap(new_words)
        if not new_words:
            return

        self.word_buffer.extend(new_words)
        self.chunk_count += 1
        self._age_candidates()

        feed_newly_emitted: set = set()
        ran_full_window = False

        while len(self.word_buffer) >= self.WINDOW_SIZE:
            window = self.word_buffer[:self.WINDOW_SIZE]
            yield from self._process_window(window, feed_newly_emitted)
            self.word_buffer = self.word_buffer[self.STRIDE:]
            ran_full_window = True

        MIN_PARTIAL = 6
        if not ran_full_window and len(self.word_buffer) >= MIN_PARTIAL:
            yield from self._process_window(self.word_buffer, feed_newly_emitted)

    def _dedup_overlap(self, new_words: list) -> list:
        if not self.word_buffer:
            return new_words
        overlap_len = 0
        for i in range(min(len(self.word_buffer), len(new_words)), 0, -1):
            if self.word_buffer[-i:] == new_words[:i]:
                overlap_len = i
                break
        return new_words[overlap_len:]

    def _age_candidates(self):
        self.candidates = [c for c in self.candidates if c.age < self.MAX_CANDIDATE_AGE]
        for c in self.candidates:
            c.age += 1

    def _process_window(self, window: list, feed_newly_emitted: set) -> list:
        results = []
        window_text = " ".join(window)
        newly_emitted: set = set()

        # Don't attempt matching on very short windows — too noisy
        if len(window) < self.MIN_MATCH_WORDS:
            return results

        # Recall
        recalled = self.index.recall(window, top_k=self.MAX_CANDIDATES)
        existing_refs = {c.entry.reference for c in self.candidates}

        for entry, hit_count in recalled:
            if hit_count < self.MIN_RECALL_HITS:
                continue
            if entry.reference in existing_refs:
                for c in self.candidates:
                    if c.entry.reference == entry.reference:
                        c.recall_score = max(c.recall_score, hit_count)
                        c.age = 0
            else:
                self.candidates.append(Candidate(entry=entry, recall_score=hit_count))
                existing_refs.add(entry.reference)

        self.candidates.sort(key=lambda c: c.recall_score, reverse=True)
        self.candidates = self.candidates[:self.MAX_CANDIDATES]

        # Fuzzy confirm — use best-subwindow overlap to handle Whisper garble
        for candidate in self.candidates:
            score = partial_ratio(window_text, candidate.entry.norm_text)
            # Use best-subwindow overlap: slide a sub-window the size of the
            # verse over the chunk and take the best overlap found.
            # This handles "...john 3:16 text... [garbled whisper tail]" correctly.
            overlap = self._best_subwindow_overlap(window, candidate.entry)
            candidate.confirm_score = max(candidate.confirm_score, score)

            if score >= self.PROMOTE_THRESHOLD and overlap >= self.MIN_CONFIRM_OVERLAP:
                if candidate.entry.reference not in self.emitted:
                    results.append(self._make_result(candidate, score, False))
                    self.emitted.add(candidate.entry.reference)
                    newly_emitted.add(candidate.entry.reference)
                    feed_newly_emitted.add(candidate.entry.reference)
                candidate.confirmed = True

            elif score >= self.CONFIRM_THRESHOLD and overlap >= self.MIN_CONFIRM_OVERLAP:
                candidate.confirmed = True

        # Emit confirmed
        for candidate in self.candidates:
            if candidate.confirmed and candidate.entry.reference not in self.emitted:
                current_overlap = self._best_subwindow_overlap(window, candidate.entry)
                if current_overlap >= self.MIN_CONFIRM_OVERLAP:
                    results.append(self._make_result(candidate, candidate.confirm_score, False))
                    self.emitted.add(candidate.entry.reference)
                    newly_emitted.add(candidate.entry.reference)
                    feed_newly_emitted.add(candidate.entry.reference)

        # Continuation
        for candidate in self.candidates:
            ref = candidate.entry.reference
            if (ref in self.emitted
                    and ref not in newly_emitted
                    and ref not in feed_newly_emitted):
                score = partial_ratio(window_text, candidate.entry.norm_text)
                overlap = self._best_subwindow_overlap(window, candidate.entry)
                if score >= self.CONTINUATION_THRESHOLD and overlap >= self.CONTINUATION_OVERLAP:
                    results.append(self._make_result(candidate, score, True))

        # Dedup: when EMIT_BEST_VERSION_ONLY is set, keep only the highest-scoring
        # version per book/chapter/verse so the UI doesn't get flooded with
        # ESV + KJV + NIV + ASV + NLT all firing for the same verse.
        if self.EMIT_BEST_VERSION_ONLY and results:
            verse_best: dict = {}  # "book chapter:verse is_cont" → result
            for r in results:
                key = f"{r['book'].lower()} {r['chapter']}:{r['verse']} {r['is_continuation']}"
                if key not in verse_best or r['score'] > verse_best[key]['score']:
                    verse_best[key] = r
            results = list(verse_best.values())

        return results

    def _content_word_overlap(self, window: list, entry: VerseEntry) -> float:
        content = [w for w in window if w not in STOPWORDS]
        if not content:
            return 0.0
        verse_words = set(entry.norm_words)
        return sum(1 for w in content if w in verse_words) / len(content)

    def _best_subwindow_overlap(self, window: list, entry: VerseEntry) -> float:
        """
        Returns the best content-word overlap found in any sub-window of
        the chunk sized to the verse length.

        This handles Whisper garble at chunk boundaries:
            "bigger insulin [john 3:16 words] lewis lover"
        The sub-window centred on the verse portion scores high even
        though the full-window overlap is dragged down by noise words.

        Returns max(full_window_overlap, best_subwindow_overlap).
        """
        # Full window overlap first (fast path — often sufficient)
        full_overlap = self._content_word_overlap(window, entry)

        # Only bother sliding if full overlap is borderline
        if full_overlap >= self.MIN_CONFIRM_OVERLAP:
            return full_overlap

        verse_words = set(entry.norm_words)
        # Use ALL window words for sliding (not just content) so sub-window
        # size matches verse length naturally
        verse_len = len(entry.norm_words)
        if verse_len >= len(window):
            return full_overlap   # verse longer than window, no sliding needed

        best = full_overlap
        for i in range(len(window) - verse_len + 1):
            sub = window[i:i + verse_len]
            content_sub = [w for w in sub if w not in STOPWORDS]
            if not content_sub:
                continue
            score = sum(1 for w in content_sub if w in verse_words) / len(content_sub)
            if score > best:
                best = score

        return best

    def _make_result(self, candidate: Candidate, score: float, is_continuation: bool) -> dict:
        e = candidate.entry
        return {
            "score": round(score / 100, 4),
            "ref": e.reference,
            "text": e.raw_text,
            "book": e.book,
            "chapter": e.chapter,
            "verse": e.verse,
            "version": e.version,
            "reference": e.reference,
            "is_continuation": is_continuation,
        }


# ============================================================================
# 4.  VERSE LOADER
# ============================================================================

VERSION_FILES = {
    "ASV": "ASV_bible.json",
    "ESV": "ESV_bible.json",
    "KJV": "KJV_bible.json",
    "NIV": "NIV_bible.json",
    "NLT": "NLT_bible.json",
}


def load_verses_from_json(filepath: str, version: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    verses = []
    for book, chapters in data.items():
        for chapter_str, verse_dict in chapters.items():
            for verse_str, text in verse_dict.items():
                if not isinstance(text, str) or not text.strip():
                    continue
                chapter = int(chapter_str)
                verse = int(verse_str)
                reference = f"{book} {chapter}:{verse} ({version})"
                verses.append({
                    "reference": reference,
                    "book": book,
                    "chapter": chapter,
                    "verse": verse,
                    "version": version,
                    "text": text.strip()
                })
    return verses


def build_index_from_verse_list(verse_dicts: list) -> VerseIndex:
    index = VerseIndex()
    for v in verse_dicts:
        try:
            entry = build_verse_entry(v)
            index.add(entry)
        except Exception as ex:
            print(f"Skipping verse {v.get('reference', '?')}: {ex}")
    print(f"✅ VerseIndex built: {len(index.all_verses):,} verses indexed")
    return index


# ============================================================================
# 5.  BIBLESEARCH  — drop-in replacement class
# ============================================================================

class BibleSearch:
    """
    Drop-in replacement for the original BibleSearch class.

    Replaces the BM25 → semantic rerank → ONNX cross-encoder pipeline with
    a direct n-gram + phonetic inverted index + fuzzy alignment approach.

    Public API is identical:
        search(query, ...)        → list of result dicts
        get_stream_matcher()      → BibleStreamMatcher for transcription loop
        get_all_verses()          → flat list of all verse dicts (for reference)
    """

    def __init__(self, device_type="cpu"):
        # device_type kept for API compatibility — not used in new pipeline
        self.device_type = device_type

        self.BIBLE_DIR = resource_path("bibles")
        self.FINAL_TOP_K = 10

        self._verse_list: list = []
        self._index: VerseIndex = None
        self._stream_matcher: BibleStreamMatcher = None

        self._load()

    def _load(self):
        print("📖 Loading Bible search data...")
        t0 = time.time()

        all_verses = []
        for version, filename in VERSION_FILES.items():
            filepath = os.path.join(self.BIBLE_DIR, filename)
            if not os.path.exists(filepath):
                print(f"⚠️  Skipping {version} — not found: {filepath}")
                continue
            verses = load_verses_from_json(filepath, version)
            print(f"  {version}: {len(verses):,} verses")
            all_verses.extend(verses)

        self._verse_list = all_verses
        print(f"  Total: {len(all_verses):,} verses loaded in {time.time() - t0:.1f}s")

        print("⚙️  Building verse index...")
        t1 = time.time()
        self._index = build_index_from_verse_list(all_verses)
        print(f"  Index built in {time.time() - t1:.1f}s")

        self._stream_matcher = BibleStreamMatcher(self._index)
        print("✅ Bible search initialized successfully!")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, bm25_top_k=None, semantic_top_k=None, final_top_k=None) -> list:
        """
        Search for Bible verses matching the query.

        Parameters kept for API compatibility — bm25_top_k and semantic_top_k
        are unused in the new pipeline. final_top_k caps result count.

        Returns list of dicts:
            reference, book, chapter, verse, version, text, score
        """
        print(f"\n🔍 Searching for: '{query}'")
        t0 = time.time()

        top_k = final_top_k or self.FINAL_TOP_K
        query_words = normalize(query).split()

        if not query_words:
            return []

        # Stage 1: n-gram recall
        recalled = self._index.recall(query_words, top_k=top_k * 5)

        if not recalled:
            print("  No search results found")
            return []

        # Stage 2: fuzzy confirm + score
        query_norm = normalize(query)
        scored = []
        for entry, hit_count in recalled:
            score = partial_ratio(query_norm, entry.norm_text)
            overlap = self._content_word_overlap(query_words, entry)
            # Combined score: fuzzy match weighted by recall hits and word overlap
            combined = (score * 0.7) + (min(hit_count, 20) * 1.0) + (overlap * 20)
            scored.append((entry, score, combined))

        # Sort by combined score, deduplicate by book+chapter+verse keeping best version
        scored.sort(key=lambda x: x[2], reverse=True)

        verse_map: dict = {}
        for entry, score, combined in scored:
            verse_key = f"{entry.book.lower()} {entry.chapter}:{entry.verse}"
            if verse_key not in verse_map or combined > verse_map[verse_key][2]:
                verse_map[verse_key] = (entry, score, combined)

        top_results = sorted(verse_map.values(), key=lambda x: x[2], reverse=True)[:top_k]

        results = []
        for entry, score, _ in top_results:
            results.append({
                "score": round(score / 100, 4),   # float 0.0–1.0, matches cross-encoder output
                "ref": entry.reference,            # original key name callers expect
                "text": entry.raw_text,
                "book": entry.book,
                "chapter": entry.chapter,
                "verse": entry.verse,
                "version": entry.version,
                "reference": entry.reference,      # bonus key for any new callers
            })

        print(f"  Search completed in {time.time() - t0:.4f}s — {len(results)} results")
        return results

    def get_stream_matcher(self) -> BibleStreamMatcher:
        """
        Returns the BibleStreamMatcher instance built on the same index.
        Use this in your transcription loop instead of calling search() per chunk.
        """
        return self._stream_matcher

    def get_all_verses(self) -> list:
        """Returns the full flat verse list (for reference or export)."""
        return self._verse_list

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _content_word_overlap(self, query_words: list, entry: VerseEntry) -> float:
        content = [w for w in query_words if w not in STOPWORDS]
        if not content:
            return 0.0
        verse_words = set(entry.norm_words)
        return sum(1 for w in content if w in verse_words) / len(content)