# bible_whisper_stream.py
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
import threading
import time

# Optional fast deps
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

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","they","them",
    "their","he","him","his","she","her","it","its","we","us","our","you","your",
    "i","my","me","who","that","this","these","those","not","no","so","as","if",
    "by","from","up","out","then","than","now","all","one","two","said","say",
    "says","again","also","just","when","what"
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
    if not w: return ""
    return w[0] + re.sub(r"[AEIOU]", "", w[1:])

def words_to_phonetic(words: list) -> list:
    return [phonetic_key(w) for w in words if w]

def partial_ratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return _rfuzz.partial_ratio(a, b)
    if len(a) > len(b): a,b = b,a
    best = 0.0
    for i in range(len(b) - len(a) + 1):
        r = difflib.SequenceMatcher(None, a, b[i:i+len(a)]).ratio() * 100
        if r > best: best = r
    return best

# ---------------------------------------------------------------------------
# Verse structures
# ---------------------------------------------------------------------------
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
    raw = verse_dict.get("text","")
    norm = normalize(raw)
    words = norm.split()
    phones = words_to_phonetic(words)
    def make_ngrams(seq,n=3):
        return set(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return VerseEntry(
        reference=verse_dict["reference"],
        book=verse_dict.get("book",""),
        chapter=int(verse_dict.get("chapter",0)),
        verse=int(verse_dict.get("verse",0)),
        version=verse_dict.get("version",""),
        raw_text=raw,
        norm_text=norm,
        norm_words=words,
        phonetic_words=phones,
        ngrams=make_ngrams(words),
        phonetic_ngrams=make_ngrams(phones)
    )

class VerseIndex:
    def __init__(self):
        self.text_index = defaultdict(list)
        self.phone_index = defaultdict(list)
        self.all_verses = []

    def add(self, entry: VerseEntry):
        self.all_verses.append(entry)
        for ng in entry.ngrams: self.text_index[ng].append(entry)
        for ng in entry.phonetic_ngrams: self.phone_index[ng].append(entry)

    def recall(self, query_words: list, top_k=20) -> list:
        query_phones = words_to_phonetic(query_words)
        def make_ngrams(seq,n=3): return set(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
        text_ngs = make_ngrams(query_words)
        phone_ngs = make_ngrams(query_phones)
        scores = {}
        for ng in text_ngs:
            for entry in self.text_index.get(ng,[]):
                scores.setdefault(entry.reference,[entry,0])[1] += 2
        for ng in phone_ngs:
            for entry in self.phone_index.get(ng,[]):
                scores.setdefault(entry.reference,[entry,0])[1] += 1
        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [(e,s) for e,s in ranked[:top_k]]

# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    entry: VerseEntry
    recall_score: int
    confirm_score: float=0.0
    age: int=0
    confirmed: bool=False

# ---------------------------------------------------------------------------
# Stream matcher
# ---------------------------------------------------------------------------
class BibleStreamMatcher:
    WINDOW_SIZE = 12
    STRIDE = 4
    MIN_RECALL_HITS = 1
    CONFIRM_THRESHOLD = 72.0
    PROMOTE_THRESHOLD = 85.0
    MIN_CONFIRM_OVERLAP = 0.25
    CONTINUATION_THRESHOLD = 72.0
    CONTINUATION_OVERLAP = 0.45
    MAX_CANDIDATE_AGE = 8
    MAX_CANDIDATES = 15

    def __init__(self, verse_index: VerseIndex):
        self.index = verse_index
        self.reset()

    def reset(self):
        self.word_buffer = []
        self.candidates = []
        self.emitted = set()
        self.chunk_count = 0

    def feed(self, raw_text: str):
        norm = normalize(raw_text)
        new_words = norm.split()
        if not new_words: return
        new_words = self._dedup_overlap(new_words)
        if not new_words: return
        self.word_buffer.extend(new_words)
        self.chunk_count += 1
        self._age_candidates()
        feed_newly_emitted = set()
        ran_full_window = False
        while len(self.word_buffer) >= self.WINDOW_SIZE:
            window = self.word_buffer[:self.WINDOW_SIZE]
            yield from self._process_window(window, feed_newly_emitted)
            self.word_buffer = self.word_buffer[self.STRIDE:]
            ran_full_window = True
        MIN_PARTIAL = 6
        if not ran_full_window and len(self.word_buffer) >= MIN_PARTIAL:
            yield from self._process_window(self.word_buffer, feed_newly_emitted)

    def _dedup_overlap(self,new_words:list)->list:
        if not self.word_buffer: return new_words
        overlap_len = 0
        for i in range(min(len(self.word_buffer), len(new_words)),0,-1):
            if self.word_buffer[-i:] == new_words[:i]: overlap_len = i; break
        return new_words[overlap_len:]

    def _age_candidates(self):
        self.candidates = [c for c in self.candidates if c.age<self.MAX_CANDIDATE_AGE]
        for c in self.candidates: c.age += 1

    def _process_window(self, window:list, feed_newly_emitted:set):
        results = []
        window_text = " ".join(window)
        newly_emitted = set()
        recalled = self.index.recall(window, top_k=self.MAX_CANDIDATES)
        existing_refs = {c.entry.reference for c in self.candidates}
        for entry, hit_count in recalled:
            if hit_count < self.MIN_RECALL_HITS: continue
            if entry.reference in existing_refs:
                for c in self.candidates:
                    if c.entry.reference == entry.reference:
                        c.recall_score = max(c.recall_score, hit_count)
                        c.age = 0
            else:
                self.candidates.append(Candidate(entry=entry, recall_score=hit_count))
                existing_refs.add(entry.reference)
        self.candidates.sort(key=lambda c:c.recall_score,reverse=True)
        self.candidates = self.candidates[:self.MAX_CANDIDATES]

        # Fuzzy confirm
        for c in self.candidates:
            score = partial_ratio(window_text, c.entry.norm_text)
            overlap = self._content_word_overlap(window, c.entry)
            c.confirm_score = max(c.confirm_score, score)
            if score >= self.PROMOTE_THRESHOLD and overlap >= self.MIN_CONFIRM_OVERLAP:
                if c.entry.reference not in self.emitted:
                    results.append(self._make_result(c,score,is_continuation=False))
                    self.emitted.add(c.entry.reference)
                    newly_emitted.add(c.entry.reference)
                    feed_newly_emitted.add(c.entry.reference)
                c.confirmed = True
            elif score >= self.CONFIRM_THRESHOLD and overlap >= self.MIN_CONFIRM_OVERLAP:
                c.confirmed = True

        for c in self.candidates:
            if c.confirmed and c.entry.reference not in self.emitted:
                current_overlap = self._content_word_overlap(window, c.entry)
                if current_overlap >= self.MIN_CONFIRM_OVERLAP:
                    results.append(self._make_result(c,c.confirm_score,is_continuation=False))
                    self.emitted.add(c.entry.reference)
                    newly_emitted.add(c.entry.reference)
                    feed_newly_emitted.add(c.entry.reference)

        # Continuation
        for c in self.candidates:
            ref = c.entry.reference
            if (ref in self.emitted and ref not in newly_emitted and ref not in feed_newly_emitted):
                score = partial_ratio(window_text, c.entry.norm_text)
                overlap = self._content_word_overlap(window, c.entry)
                if score >= self.CONTINUATION_THRESHOLD and overlap >= self.CONTINUATION_OVERLAP:
                    results.append(self._make_result(c,score,is_continuation=True))
        return results

    def _content_word_overlap(self, window:list, entry:VerseEntry)->float:
        content = [w for w in window if w not in STOPWORDS]
        if not content: return 0.0
        verse_words = set(entry.norm_words)
        matches = sum(1 for w in content if w in verse_words)
        return matches/len(content)

    def _make_result(self, c:Candidate, score:float, is_continuation:bool)->dict:
        e = c.entry
        return {
            "reference": e.reference,
            "book": e.book,
            "chapter": e.chapter,
            "verse": e.verse,
            "version": e.version,
            "text": e.raw_text,
            "score": round(score/100,4),
            "is_continuation": is_continuation
        }

# ---------------------------------------------------------------------------
# Demo streaming integration with Whisper-like queue
# ---------------------------------------------------------------------------
def run_demo_stream():
    # Example verse index
    sample_verses = [
        {"reference":"John 3:16 (NLT)","book":"John","chapter":3,"verse":16,"version":"NLT",
         "text":"for this is how god loved the world he gave his one and only son so that everyone who believes in him will not perish but have eternal life"},
        {"reference":"Psalm 23:1 (KJV)","book":"Psalm","chapter":23,"verse":1,"version":"KJV",
         "text":"the lord is my shepherd i shall not want"},
        {"reference":"Isaiah 1:18 (KJV)","book":"Isaiah","chapter":1,"verse":18,"version":"KJV",
         "text":"come now and let us reason together saith the lord though your sins be as scarlet they shall be as white as snow though they be red like crimson they shall be as wool"}
    ]
    index = VerseIndex()
    for v in sample_verses: index.add(build_verse_entry(v))

    matcher = BibleStreamMatcher(index)

    # Simulated live transcription queue
    transcription_queue = Queue()

    def transcription_simulator():
        chunks = [
    "god is a give it says that he loved the world that he gave his one and only son",
    "everyone who believes in him will not perish but have eternal life",
    "do you not know that the lord is my shepherd i shall not want",
    "though your sins be as scarlet they shall be as white as snow",
    "come now and let us reason together saith the lord",
    "though they be red like crimson they shall be as wool",
    "for god so loved the world that he gave his only son",
    "blessed are the meek for they shall inherit the earth",
    "love your neighbor as yourself this is the commandment",
    "in the beginning god created the heavens and the earth",
    "the spirit of the lord is upon me because he hath anointed me",
    "the lord is my light and my salvation whom shall i fear",
    "i can do all things through christ which strengtheneth me",
    "trust in the lord with all thine heart and lean not unto thine own understanding",
    "be strong and of a good courage fear not neither be afraid",
    "for where your treasure is there will your heart be also",
    "ask and it shall be given you seek and ye shall find knock and it shall be opened",
    "let your light so shine before men that they may see your good works",
    "but seek ye first the kingdom of god and his righteousness",
    "for i know the thoughts that i think toward you saith the lord thoughts of peace and not of evil",
]
        for c in chunks:
            transcription_queue.put(c)
            time.sleep(0.2)

    # Start simulator in background
    threading.Thread(target=transcription_simulator, daemon=True).start()

    print("🎧 Listening for transcribed text...")
    try:
        while True:
            chunk = transcription_queue.get(timeout=5)
            for result in matcher.feed(chunk):
                print(f'input: {chunk}')
                print(f"[{result['reference']}] Score: {result['score']} Continuation: {result['is_continuation']}")
                print(f"Text: {result['text']}\n")
    except Exception as e:
        print("Stream ended or timeout.")

if __name__ == "__main__":
    run_demo_stream()