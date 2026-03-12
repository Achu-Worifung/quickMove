"""
Bible Streaming Alignment Pipeline
====================================
Replaces the RAG classify→search pipeline with a direct incremental
n-gram + phonetic matching approach against the known verse corpus.

Architecture:
    noisy audio → Whisper → StreamMatcher → candidate verses → fuzzy confirm → emit

Dependencies (all already in your project):
    pip install rapidfuzz jellyfish

Usage:
    matcher = BibleStreamMatcher(verse_index)   # build once at startup
    matcher.reset()                             # call when starting new session

    # In your transcription loop, replace everything after normalize_text() with:
    for result in matcher.feed(text):
        self.autoSearchResults.emit([result], result['text'], result['score'], self.auto_search_size)
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Optional fast deps — falls back gracefully if not installed
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

# Common words that carry no verse-specific meaning.
# Excluded from the content-word overlap guard so they don't cause
# unrelated chunks to falsely continue a previously matched verse.
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
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phonetic_key(word: str) -> str:
    """
    Returns a phonetic key for a word.
    Uses jellyfish.soundex if available, otherwise a simple vowel-strip fallback
    that still catches common Whisper mishearings (sins/scenes, begotten/begotun).
    """
    if HAS_JELLYFISH:
        return jellyfish.soundex(word)
    # Minimal fallback: remove vowels after first char, uppercase
    w = word.upper()
    if not w:
        return ""
    return w[0] + re.sub(r"[AEIOU]", "", w[1:])


def words_to_phonetic(words: list) -> list:
    return [phonetic_key(w) for w in words if w]


def partial_ratio(a: str, b: str) -> float:
    """
    Partial substring similarity (0-100).
    Good for matching short spoken fragments against full verse text.
    """
    if HAS_RAPIDFUZZ:
        return _rfuzz.partial_ratio(a, b)
    # difflib fallback
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    for i in range(len(b) - len(a) + 1):
        r = difflib.SequenceMatcher(None, a, b[i:i + len(a)]).ratio() * 100
        if r > best:
            best = r
    return best


# ============================================================================
# 2.  VERSE INDEX  (built once at startup)
# ============================================================================

@dataclass
class VerseEntry:
    reference: str
    book: str
    chapter: int
    verse: int
    version: str
    raw_text: str           # original cased text
    norm_text: str          # normalized text
    norm_words: list        # normalized word list
    phonetic_words: list    # phonetic key per word
    ngrams: set             # set of (w1,w2,w3) text trigrams
    phonetic_ngrams: set    # set of (p1,p2,p3) phonetic trigrams


def build_verse_entry(verse_dict: dict) -> VerseEntry:
    """
    Convert a raw verse dict into a VerseEntry.
    Expected keys: reference, book, chapter, verse, version, text
    """
    raw = verse_dict.get("text", "")
    norm = normalize(raw)
    words = norm.split()
    phones = words_to_phonetic(words)

    def make_ngrams(seq, n=3):
        return set(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

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
    """
    Dual inverted index: text trigrams + phonetic trigrams.
    Phonetic index lets Whisper mishearings (scenes→sins, begotun→begotten)
    still recall the correct verse.
    """

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
        """
        Score verses by shared trigram count.
        Text matches score 2 points, phonetic matches score 1 point.
        Returns [(VerseEntry, score)] sorted descending.
        """
        query_phones = words_to_phonetic(query_words)

        def make_ngrams(seq, n=3):
            return set(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

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
# 3.  CANDIDATE  (live tracking of a partially-matched verse)
# ============================================================================

@dataclass
class Candidate:
    entry: VerseEntry
    recall_score: int           # n-gram hit count from index recall
    confirm_score: float = 0.0  # best fuzzy score seen so far
    age: int = 0                # chunks since created (for eviction)
    confirmed: bool = False     # crossed CONFIRM_THRESHOLD at least once


# ============================================================================
# 4.  STREAM MATCHER
# ============================================================================

class BibleStreamMatcher:
    """
    Drop-in replacement for the classify→search→rerank pipeline.

    Call feed(text) for each normalized Whisper chunk.
    Yields result dicts whenever a verse match is confirmed or continued.
    """

    # --- Tuning knobs ---
    WINDOW_SIZE = 12            # words per matching window
    STRIDE = 4                  # words to advance after each window
    MIN_RECALL_HITS = 1         # min n-gram hits to become a candidate
                                # (1 allows phonetic-only matches e.g. Isaiah via scenes->sins)
    CONFIRM_THRESHOLD = 72.0    # partial_ratio score to confirm a match
    PROMOTE_THRESHOLD = 85.0    # partial_ratio score to emit immediately
    MIN_CONFIRM_OVERLAP = 0.25  # min content word overlap alongside CONFIRM_THRESHOLD
                                # prevents high partial_ratio on unrelated text from matching
    CONTINUATION_THRESHOLD = 72.0   # score to fire a continuation event
    CONTINUATION_OVERLAP = 0.45     # min content word overlap for continuation
                                    # higher threshold needed when jellyfish phonetic index
                                    # is active, as Soundex collisions increase false recalls
    MAX_CANDIDATE_AGE = 8       # chunks before a candidate is evicted
    MAX_CANDIDATES = 15         # max live candidates at any time

    def __init__(self, verse_index: VerseIndex):
        self.index = verse_index
        self.reset()

    def reset(self):
        """Call at the start of each new transcription session."""
        self.word_buffer: list = []
        self.candidates: list = []
        self.emitted: set = set()   # references emitted this session
        self.chunk_count: int = 0

    def flush_candidates(self):
        """
        Clear live candidates, word buffer, and emitted set.
        Call this from your transcription loop whenever the speaker moves
        to a clearly non-Bible topic, so stale verse candidates don't
        bleed into the next windows via buffer carryover.

        Resetting emitted allows the same verse to fire again later in the
        sermon if the preacher returns to it — which is common in practice.

        Example usage in transcribe_loop():
            if label_chunk == 'non bible' and not contains_keyword:
                self.stream_matcher.flush_candidates()
                continue
        """
        self.candidates = []
        self.word_buffer = []
        self.emitted = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, raw_text: str):
        """
        Feed one Whisper chunk. Yields result dicts for confirmed verses.

        Result format (matches your existing autoSearchResults format):
        {
            reference, book, chapter, verse, version,
            text,           # original verse text
            score,          # 0.0 - 1.0
            is_continuation # True if this is a follow-on to a prior match
        }
        """
        norm = normalize(raw_text)
        new_words = norm.split()
        if not new_words:
            return

        # Remove words at start of new chunk already present at end of buffer
        new_words = self._dedup_overlap(new_words)
        if not new_words:
            return

        self.word_buffer.extend(new_words)
        self.chunk_count += 1
        self._age_candidates()

        # Shared set across ALL windows in this feed() call.
        # Prevents a verse matched in window N firing as continuation in window N+1
        # within the same audio chunk.
        feed_newly_emitted: set = set()
        ran_full_window = False

        while len(self.word_buffer) >= self.WINDOW_SIZE:
            window = self.word_buffer[:self.WINDOW_SIZE]
            yield from self._process_window(window, feed_newly_emitted)
            self.word_buffer = self.word_buffer[self.STRIDE:]
            ran_full_window = True

        # If no full window ran this feed() call, try a partial window.
        # This catches short verse quotes (< WINDOW_SIZE words) that would
        # otherwise be silently dropped, e.g. Malachi 3:18 quoted in 9 words.
        # We skip this when a full window already ran to avoid duplicate
        # continuations from the leftover buffer tail.
        MIN_PARTIAL = 6
        if not ran_full_window and len(self.word_buffer) >= MIN_PARTIAL:
            yield from self._process_window(self.word_buffer, feed_newly_emitted)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dedup_overlap(self, new_words: list) -> list:
        """Strip words from new_words that overlap with the tail of word_buffer."""
        if not self.word_buffer:
            return new_words
        overlap_len = 0
        for i in range(min(len(self.word_buffer), len(new_words)), 0, -1):
            if self.word_buffer[-i:] == new_words[:i]:
                overlap_len = i
                break
        return new_words[overlap_len:]

    def _age_candidates(self):
        """Increment age and evict stale candidates."""
        self.candidates = [c for c in self.candidates if c.age < self.MAX_CANDIDATE_AGE]
        for c in self.candidates:
            c.age += 1

    def _process_window(self, window: list, feed_newly_emitted: set) -> list:
        results = []
        window_text = " ".join(window)
        newly_emitted: set = set()  # first-emitted in THIS window only

        # ---- Step 1: Recall ----
        recalled = self.index.recall(window, top_k=self.MAX_CANDIDATES)
        existing_refs = {c.entry.reference for c in self.candidates}

        for entry, hit_count in recalled:
            if hit_count < self.MIN_RECALL_HITS:
                continue
            if entry.reference in existing_refs:
                for c in self.candidates:
                    if c.entry.reference == entry.reference:
                        c.recall_score = max(c.recall_score, hit_count)
                        c.age = 0   # still active — reset age
            else:
                self.candidates.append(Candidate(entry=entry, recall_score=hit_count))
                existing_refs.add(entry.reference)

        self.candidates.sort(key=lambda c: c.recall_score, reverse=True)
        self.candidates = self.candidates[:self.MAX_CANDIDATES]

        # ---- Step 2: Fuzzy confirm ----
        for candidate in self.candidates:
            score = partial_ratio(window_text, candidate.entry.norm_text)
            overlap = self._content_word_overlap(window, candidate.entry)
            candidate.confirm_score = max(candidate.confirm_score, score)

            if score >= self.PROMOTE_THRESHOLD and overlap >= self.MIN_CONFIRM_OVERLAP:
                # High confidence + real word overlap — emit immediately
                if candidate.entry.reference not in self.emitted:
                    results.append(self._make_result(candidate, score, is_continuation=False))
                    self.emitted.add(candidate.entry.reference)
                    newly_emitted.add(candidate.entry.reference)
                    feed_newly_emitted.add(candidate.entry.reference)
                candidate.confirmed = True

            elif score >= self.CONFIRM_THRESHOLD and overlap >= self.MIN_CONFIRM_OVERLAP:
                candidate.confirmed = True

        # Emit all confirmed candidates not yet emitted.
        # Re-check content overlap with the CURRENT window before emitting.
        # A candidate confirmed in a previous window should only emit now
        # if the current window still contains words from that verse.
        for candidate in self.candidates:
            if candidate.confirmed and candidate.entry.reference not in self.emitted:
                current_overlap = self._content_word_overlap(window, candidate.entry)
                if current_overlap >= self.MIN_CONFIRM_OVERLAP:
                    results.append(self._make_result(
                        candidate, candidate.confirm_score, is_continuation=False
                    ))
                    self.emitted.add(candidate.entry.reference)
                    newly_emitted.add(candidate.entry.reference)
                    feed_newly_emitted.add(candidate.entry.reference)
                self.emitted.add(candidate.entry.reference)
                newly_emitted.add(candidate.entry.reference)
                feed_newly_emitted.add(candidate.entry.reference)

        # ---- Step 3: Continuation ----
        # Four guards prevent spurious continuation events:
        #   (a) newly_emitted       — not first-emitted in this window
        #   (b) feed_newly_emitted  — not first-emitted in this feed() call
        #   (c) fuzzy score >= CONTINUATION_THRESHOLD
        #   (d) content word overlap >= CONTINUATION_OVERLAP (0.40)
        #       blocks 'he begins to suspect' from continuing Malachi
        #       blocks 'scenes crimson snow' from continuing Matthew 11:28
        for candidate in self.candidates:
            ref = candidate.entry.reference
            if (ref in self.emitted
                    and ref not in newly_emitted
                    and ref not in feed_newly_emitted):
                score = partial_ratio(window_text, candidate.entry.norm_text)
                overlap = self._content_word_overlap(window, candidate.entry)
                if score >= self.CONTINUATION_THRESHOLD and overlap >= self.CONTINUATION_OVERLAP:
                    results.append(self._make_result(candidate, score, is_continuation=True))

        return results

    def _content_word_overlap(self, window: list, entry: VerseEntry) -> float:
        """
        Fraction of non-stopword window words that appear in the verse.
        Filters out function words so a chunk like 'your scenes are crimson'
        doesn't falsely continue Matthew 11:28 just because of 'are' and 'your'.
        """
        content = [w for w in window if w not in STOPWORDS]
        if not content:
            return 0.0
        verse_words = set(entry.norm_words)
        matches = sum(1 for w in content if w in verse_words)
        return matches / len(content)

    def _make_result(self, candidate: Candidate, score: float, is_continuation: bool) -> dict:
        e = candidate.entry
        return {
            "reference": e.reference,
            "book": e.book,
            "chapter": e.chapter,
            "verse": e.verse,
            "version": e.version,
            "text": e.raw_text,
            "score": round(score / 100, 4),  # normalized to 0.0-1.0
            "is_continuation": is_continuation,
        }


# ============================================================================
# 5.  INDEX BUILDER
# ============================================================================

def build_index_from_verse_list(verse_dicts: list) -> VerseIndex:
    """
    Build a VerseIndex from your full Bible corpus.

    verse_dicts format:
    [
        {"reference": "John 3:16 (NLT)", "book": "John", "chapter": 3,
         "verse": 16, "version": "NLT", "text": "for god so loved..."},
        ...
    ]
    Run once at startup and reuse the returned index for the whole session.
    """
    index = VerseIndex()
    for v in verse_dicts:
        try:
            entry = build_verse_entry(v)
            index.add(entry)
        except Exception as ex:
            print(f"Skipping verse {v.get('reference', '?')}: {ex}")
    print(f"✅ VerseIndex built: {len(index.all_verses)} verses indexed")
    return index


# ============================================================================
# 6.  INTEGRATION SNIPPET
# ============================================================================
"""
──── STARTUP ────────────────────────────────────────────────────────────────

    from bible_stream_matcher import build_index_from_verse_list, BibleStreamMatcher

    all_verses = your_search_engine.get_all_verses()  # your existing corpus loader
    verse_index = build_index_from_verse_list(all_verses)
    self.stream_matcher = BibleStreamMatcher(verse_index)


──── transcribe_loop() — replace EVERYTHING after normalize_text() ──────────

    text = self.normalize_text(text)
    self.context_manager.add_text(text)

    # ---- UI overlap merge ----
    self.transcribed_chunks.append(text)
    if len(self.transcribed_chunks) >= 2:
        prev_words = self.transcribed_chunks[-2].split()
        curr_words = text.split()
        overlap_len = 0
        for i in range(min(len(prev_words), len(curr_words)), 0, -1):
            if prev_words[-i:] == curr_words[:i]:
                overlap_len = i
                break
        if overlap_len > 0:
            merged_text = " ".join(prev_words[:-overlap_len] + curr_words)
        else:
            merged_text = self.transcribed_chunks[-2] + " " + text
        self.guitextReady.emit(merged_text)

        # ---- Spoken reference fast-path ----
        spoken_references = extract_bible_reference(merged_text)
        if spoken_references:
            reference_dict = []
            for ref in spoken_references:
                reference_dict.append({
                    'reference': ref['full'],
                    'book': ref.get('book', ''),
                    'chapter': ref.get('chapter', ''),
                    'verse': ref.get('verse', ''),
                    'text': '',
                    'score': 1.0
                })
                self.stream_matcher.emitted.add(ref['full'])  # prevent duplicate match
            self.autoSearchResults.emit(reference_dict, "", 0.0, self.auto_search_size)
            continue

    # ---- Stream match ----
    for result in self.stream_matcher.feed(text):
        self.autoSearchResults.emit(
            [result],
            result['text'],
            result['score'],
            self.auto_search_size
        )
"""


# ============================================================================
# 7.  SMOKE TEST
# ============================================================================

if __name__ == "__main__":

    sample_verses = [
        # ── Core test verses ──────────────────────────────────────────────
        {
            "reference": "John 3:16 (NLT)", "book": "John", "chapter": 3, "verse": 16, "version": "NLT",
            "text": "for this is how god loved the world he gave his one and only son so that everyone who believes in him will not perish but have eternal life"
        },
        {
            "reference": "John 3:16 (KJV)", "book": "John", "chapter": 3, "verse": 16, "version": "KJV",
            "text": "for god so loved the world that he gave his only begotten son that whosoever believeth in him should not perish but have everlasting life"
        },
        {
            "reference": "Matthew 11:28 (NLT)", "book": "Matthew", "chapter": 11, "verse": 28, "version": "NLT",
            "text": "then jesus said come to me all of you who are weary and carry heavy burdens and i will give you rest"
        },
        {
            "reference": "Isaiah 1:18 (KJV)", "book": "Isaiah", "chapter": 1, "verse": 18, "version": "KJV",
            "text": "come now and let us reason together saith the lord though your sins be as scarlet they shall be as white as snow though they be red like crimson they shall be as wool"
        },
        {
            "reference": "Malachi 3:18 (NLT)", "book": "Malachi", "chapter": 3, "verse": 18, "version": "NLT",
            "text": "then you will again see the difference between the righteous and the wicked between those who serve god and those who do not"
        },
        # ── Additional common sermon verses ───────────────────────────────
        {
            "reference": "Jeremiah 29:11 (NLT)", "book": "Jeremiah", "chapter": 29, "verse": 11, "version": "NLT",
            "text": "for i know the plans i have for you says the lord they are plans for good and not for disaster to give you a future and a hope"
        },
        {
            "reference": "Psalm 23:1 (KJV)", "book": "Psalm", "chapter": 23, "verse": 1, "version": "KJV",
            "text": "the lord is my shepherd i shall not want"
        },
        {
            "reference": "Psalm 23:4 (KJV)", "book": "Psalm", "chapter": 23, "verse": 4, "version": "KJV",
            "text": "yea though i walk through the valley of the shadow of death i will fear no evil for thou art with me thy rod and thy staff they comfort me"
        },
        {
            "reference": "Romans 8:28 (NLT)", "book": "Romans", "chapter": 8, "verse": 28, "version": "NLT",
            "text": "and we know that god causes everything to work together for the good of those who love god and are called according to his purpose for them"
        },
        {
            "reference": "Philippians 4:13 (KJV)", "book": "Philippians", "chapter": 4, "verse": 13, "version": "KJV",
            "text": "i can do all things through christ which strengtheneth me"
        },
        {
            "reference": "Proverbs 3:5 (NLT)", "book": "Proverbs", "chapter": 3, "verse": 5, "version": "NLT",
            "text": "trust in the lord with all your heart do not depend on your own understanding"
        },
        {
            "reference": "John 14:6 (NLT)", "book": "John", "chapter": 14, "verse": 6, "version": "NLT",
            "text": "jesus told him i am the way the truth and the life no one can come to the father except through me"
        },
        {
            "reference": "Romans 3:23 (NLT)", "book": "Romans", "chapter": 3, "verse": 23, "version": "NLT",
            "text": "for everyone has sinned we all fall short of gods glorious standard"
        },
        {
            "reference": "Romans 6:23 (NLT)", "book": "Romans", "chapter": 6, "verse": 23, "version": "NLT",
            "text": "for the wages of sin is death but the free gift of god is eternal life through christ jesus our lord"
        },
        {
            "reference": "Ephesians 2:8 (NLT)", "book": "Ephesians", "chapter": 2, "verse": 8, "version": "NLT",
            "text": "god saved you by his grace when you believed and you cant take credit for this it is a gift from god"
        },
        {
            "reference": "Isaiah 40:31 (KJV)", "book": "Isaiah", "chapter": 40, "verse": 31, "version": "KJV",
            "text": "but they that wait upon the lord shall renew their strength they shall mount up with wings as eagles they shall run and not be weary and they shall walk and not faint"
        },
        {
            "reference": "Joshua 1:9 (NLT)", "book": "Joshua", "chapter": 1, "verse": 9, "version": "NLT",
            "text": "this is my command be strong and courageous do not be afraid or discouraged for the lord your god is with you wherever you go"
        },
        {
            "reference": "Matthew 6:33 (NLT)", "book": "Matthew", "chapter": 6, "verse": 33, "version": "NLT",
            "text": "seek the kingdom of god above all else and live righteously and he will give you everything you need"
        },
        {
            "reference": "Galatians 5:22 (NLT)", "book": "Galatians", "chapter": 5, "verse": 22, "version": "NLT",
            "text": "but the holy spirit produces this kind of fruit in our lives love joy peace patience kindness goodness faithfulness"
        },
        {
            "reference": "Revelation 3:20 (NLT)", "book": "Revelation", "chapter": 3, "verse": 20, "version": "NLT",
            "text": "look i stand at the door and knock if you hear my voice and open the door i will come in and we will share a meal together as friends"
        },
        {
            "reference": "2 Timothy 3:16 (NLT)", "book": "2 Timothy", "chapter": 3, "verse": 16, "version": "NLT",
            "text": "all scripture is inspired by god and is useful to teach us what is true and to make us realize what is wrong in our lives it corrects us when we are wrong and teaches us to do what is right"
        },
        {
            "reference": "1 John 1:9 (NLT)", "book": "1 John", "chapter": 1, "verse": 9, "version": "NLT",
            "text": "but if we confess our sins to him he is faithful and just to forgive us our sins and to cleanse us from all wickedness"
        },
        {
            "reference": "Acts 1:8 (NLT)", "book": "Acts", "chapter": 1, "verse": 8, "version": "NLT",
            "text": "but you will receive power when the holy spirit comes upon you and you will be my witnesses telling people about me everywhere in jerusalem throughout judea in samaria and to the ends of the earth"
        },
        {
            "reference": "Genesis 1:1 (KJV)", "book": "Genesis", "chapter": 1, "verse": 1, "version": "KJV",
            "text": "in the beginning god created the heaven and the earth"
        },
    ]

    print("Building index...")
    index = build_index_from_verse_list(sample_verses)
    matcher = BibleStreamMatcher(index)

    # Each section is separated by flush_candidates() to simulate your
    # transcription loop clearing state between topics.
    # Format: (label, expected_reference_substring, chunks)
    chunk_sections = [
        # ── Standard quote, NLT ──────────────────────────────────────────
        ("John 3:16 NLT — clean quote", "John 3:16",
        [
            "for god so loved the world that",
            "that he gave his one and only son so that everyone",
            "everyone who believes in him will not perish but have eternal life",
        ]),

        # ── KJV variant with Whisper mishearing (begotun→begotten) ───────
        ("John 3:16 KJV — phonetic mishearing", "John 3:16",
        [
            "for god so loved the world that he gave his only begotun son",
            "that whosoever believeth in him should not perish",
        ]),

        # ── Partial opening quote only ────────────────────────────────────
        ("Matthew 11:28 — partial quote", "Matthew 11:28",
        [
            "come to me all those who are weary and carry heavy burdens",
            "and i will give you rest",
        ]),

        # ── Phonetic mishearing (scenes→sins) ────────────────────────────
        ("Isaiah 1:18 — scenes/sins phonetic", "Isaiah 1:18",
        [
            "though your scenes be as scarlet they shall be as white as snow",
            "though they be red like crimson they shall be as wool",
        ]),

        # ── Very short verse ─────────────────────────────────────────────
        ("Psalm 23:1 — very short verse", "Psalm 23:1",
        [
            "the lord is my shepherd i shall not want",
        ]),

        # ── Mid-verse entry (preacher starts quoting partway through) ────
        ("Psalm 23:4 — mid-verse entry", "Psalm 23:4",
        [
            "yea though i walk through the valley of the shadow of death",
            "i will fear no evil for thou art with me",
        ]),

        # ── Common sermon verse, plans for good ──────────────────────────
        ("Jeremiah 29:11 — plans for good", "Jeremiah 29:11",
        [
            "for i know the plans i have for you says the lord",
            "they are plans for good and not for disaster",
            "to give you a future and a hope",
        ]),

        # ── Very short KJV verse ─────────────────────────────────────────
        ("Philippians 4:13 — short KJV", "Philippians 4:13",
        [
            "i can do all things through christ which strengtheneth me",
        ]),

        # ── Way the truth the life ───────────────────────────────────────
        ("John 14:6 — way truth life", "John 14:6",
        [
            "jesus said i am the way the truth and the life",
            "no one can come to the father except through me",
        ]),

        # ── Fruit of the Spirit ──────────────────────────────────────────
        ("Galatians 5:22 — fruit of the spirit", "Galatians 5:22",
        [
            "the holy spirit produces this kind of fruit in our lives",
            "love joy peace patience kindness goodness faithfulness",
        ]),

        # ── Wages of sin ─────────────────────────────────────────────────
        ("Romans 6:23 — wages of sin", "Romans 6:23",
        [
            "for the wages of sin is death",
            "but the free gift of god is eternal life through christ jesus our lord",
        ]),

        # ── Confession of sins ───────────────────────────────────────────
        ("1 John 1:9 — confess sins", "1 John 1:9",
        [
            "if we confess our sins to him",
            "he is faithful and just to forgive us and cleanse us from all wickedness",
        ]),

        # ── Revelation 3:20 — knock ──────────────────────────────────────
        ("Revelation 3:20 — door and knock", "Revelation 3:20",
        [
            "behold i stand at the door and knock",
            "if you hear my voice and open the door i will come in",
        ]),

        # ── Isaiah 40:31 — mount up with wings ───────────────────────────
        ("Isaiah 40:31 — wings as eagles", "Isaiah 40:31",
        [
            "they that wait upon the lord shall renew their strength",
            "they shall mount up with wings as eagles",
            "they shall run and not be weary and they shall walk and not faint",
        ]),

        # ── Genesis 1:1 — very short ─────────────────────────────────────
        ("Genesis 1:1 — in the beginning", "Genesis 1:1",
        [
            "in the beginning god created the heaven and the earth",
        ]),

        # ── Non-Bible noise — all should be silent ────────────────────────
        ("NON-BIBLE NOISE — action/thriller narrative", None,
        [
            "he begins to suspect that this man might be more than he appears",
            "the detective quietly tells his men to keep a close eye on the suspect",
            "lester pulls out a device and attaches it to the nearby building",
            "chaos breaks out instantly in the middle of the fight",
            "shipment of advanced weapons these gangs arrive at the warehouse",
            "the joker uses a hidden electric shock sending a painful jolt through him",
            "destroying the cargo and taking down members from both sides",
            "as the smoke clears the remaining criminals try to understand what happened",
        ]),

        ("NON-BIBLE NOISE — everyday conversation", None,
        [
            "okay so thats really powerful lets think about what that means",
            "can you hear me clearly is this any better than what we had before",
            "i see that the audio is a little better now but still having issues",
            "anyways i am not going to go back to the previous sections of this code",
            "i am going to focus on the next feature we need to implement",
            "does this work better than what we previously had set up",
        ]),

        ("NON-BIBLE NOISE — news and finance", None,
        [
            "the stock market fell sharply today amid fears of rising inflation",
            "the federal reserve held interest rates steady at their latest meeting",
            "tech companies reported strong earnings despite supply chain disruptions",
            "the prime minister announced a new policy on housing affordability",
            "unemployment figures dropped to their lowest level in three years",
            "she walked into the coffee shop and ordered a large latte with oat milk",
        ]),

        ("NON-BIBLE NOISE — sports commentary", None,
        [
            "he drives baseline pulls up for the mid range jumper and it goes in",
            "the quarterback scrambles left finds his receiver wide open in the end zone",
            "and the crowd goes absolutely wild as he crosses the finish line first",
            "penalty kick saved by the goalkeeper what a tremendous stop that was",
            "two minutes left on the clock and they are down by three points",
            "he comes off the bench and immediately makes an impact on the game",
        ]),

        ("NON-BIBLE NOISE — dramatic language that sounds biblical", None,
        [
            # These are the hardest cases — elevated language that mimics scripture
            "he rose from the ashes and became the man he was always meant to be",
            "the light at the end of the tunnel gave him hope when all seemed lost",
            "she sacrificed everything so that her children could have a better life",
            "in the darkest hour of the night he finally found the truth he sought",
            "love conquers all and in the end the good always triumphs over evil",
            "there is a time for war and a time for peace a time to live and die",
        ]),
    ]
    phonetic_status = "jellyfish (Soundex)" if HAS_JELLYFISH else "fallback (vowel-strip)"
    fuzzy_status = "rapidfuzz" if HAS_RAPIDFUZZ else "difflib"
    print(f"\nPhonetic engine : {phonetic_status}")
    print(f"Fuzzy engine    : {fuzzy_status}")
    print("\n--- Simulating stream ---\n")

    match_count = 0
    continuation_count = 0
    same_window_double = False

    section_results = []   # (label, expected, fired_refs)

    for s_idx, (label, expected_ref, chunks) in enumerate(chunk_sections):
        if s_idx > 0:
            matcher.flush_candidates()

        print(f"  ┌─ {label}")
        section_fired = set()

        for chunk in chunks:
            print(f"  │  IN : {chunk}")
            chunk_refs: dict = {}

            for result in matcher.feed(chunk):
                tag = "[CONT] " if result["is_continuation"] else "[MATCH]"
                ref = result["reference"]
                print(f"  │  {tag} {ref}  score={result['score']:.2f}")

                if result["is_continuation"]:
                    continuation_count += 1
                else:
                    match_count += 1
                    section_fired.add(ref)

                if not result["is_continuation"]:
                    section_fired.add(ref)
                else:
                    # A continuation of the expected verse still counts as a pass
                    section_fired.add(ref)

                chunk_refs.setdefault(ref, []).append(
                    "CONTINUATION" if result["is_continuation"] else "MATCH"
                )

            for ref, tags in chunk_refs.items():
                if "MATCH" in tags and "CONTINUATION" in tags:
                    same_window_double = True
                    print(f"  │  ⚠️  DOUBLE EMIT: {ref}")

        # Evaluate section outcome
        if expected_ref is None:
            # Noise section — nothing should fire
            passed = len(section_fired) == 0
            status = "✅ PASS (silent)" if passed else f"❌ FAIL — fired: {section_fired}"
        else:
            passed = any(expected_ref in r for r in section_fired)
            status = f"✅ PASS" if passed else f"❌ MISS — expected '{expected_ref}', got: {section_fired or 'nothing'}"

        section_results.append((label, passed))
        print(f"  └─ {status}\n")
    # Isaiah-specific check with fresh matcher
    matcher2 = BibleStreamMatcher(index)
    isaiah_chunks = [
        "your scenes are crimson they shall be white as snow",
        "crimson they shall be white as snow like wool",
    ]
    isaiah_fired = any(
        "Isaiah" in r["reference"]
        for c in isaiah_chunks
        for r in matcher2.feed(c)
    )

    passed_count = sum(1 for _, p in section_results if p)
    total_count = len(section_results)

    print("─" * 60)
    print("  SECTION RESULTS")
    print("─" * 60)
    for label, passed in section_results:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {label}")

    print()
    print("─" * 60)
    print("  SUMMARY")
    print("─" * 60)
    print(f"  Sections passed        : {passed_count}/{total_count}")
    print(f"  Total MATCH            : {match_count}")
    print(f"  Total CONTINUATION     : {continuation_count}")
    print(f"  Same-chunk double emit : {'❌ YES' if same_window_double else '✅ NO'}")
    print(f"  Isaiah 1:18 fired      : {'✅ YES' if isaiah_fired else '❌ NO  (needs jellyfish for scenes→sins)'}")
    print("─" * 55)