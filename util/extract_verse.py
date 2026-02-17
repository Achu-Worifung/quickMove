import re
from text2digits import text2digits

BIBLE_BOOKS = {
    # Numbered books
    "1 samuel", "2 samuel", "1 kings", "2 kings",
    "1 chronicles", "2 chronicles", "1 corinthians", "2 corinthians",
    "1 thessalonians", "2 thessalonians", "1 timothy", "2 timothy",
    "1 peter", "2 peter", "1 john", "2 john", "3 john",
    "1 maccabees", "2 maccabees",
    # Single-word books
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy",
    "joshua", "judges", "ruth", "ezra", "nehemiah", "esther",
    "job", "psalms", "psalm", "proverbs", "ecclesiastes",
    "isaiah", "jeremiah", "lamentations", "ezekiel", "daniel",
    "hosea", "joel", "amos", "obadiah", "jonah", "micah",
    "nahum", "habakkuk", "zephaniah", "haggai", "zechariah", "malachi",
    "matthew", "mark", "luke", "john", "acts", "romans",
    "galatians", "ephesians", "philippians", "colossians",
    "philemon", "hebrews", "james", "jude", "revelation",
}

_t2d = text2digits.Text2Digits()

def _convert_word_numbers(text: str) -> str:
    """
    Convert written-out numbers to digits, but ONLY in segments that
    don't already contain bare digit tokens. This prevents text2digits
    from merging things like '24 2' into '242'.
    
    Strategy: split on existing digit-containing tokens, convert only
    the purely alphabetic segments, then rejoin.
    """
    # Split into alternating non-digit / digit spans
    # We convert only the non-digit spans
    segments = re.split(r'(\b\d+\b)', text)
    converted = []
    for seg in segments:
        if re.fullmatch(r'\d+', seg.strip()):
            # Already a number — leave it alone
            converted.append(seg)
        else:
            converted.append(_t2d.convert(seg))
    return ''.join(converted)

def normalize_text(text: str) -> str:
    text = _convert_word_numbers(text)
    # Remove ordinal suffixes: 1st → 1
    text = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', text, flags=re.IGNORECASE)
    # "chapter X verse Y" → "X:Y"
    text = re.sub(r'chapter\s+(\d+)\s+verse\s+(\d+)', r'\1:\2', text, flags=re.IGNORECASE)
    # "chapter X" → "X"
    text = re.sub(r'chapter\s+(\d+)', r'\1', text, flags=re.IGNORECASE)
    # "verse Y" → ":Y"
    text = re.sub(r'verse\s+(\d+)', r':\1', text, flags=re.IGNORECASE)
    return text

def find_book_in_parts(parts: list[str]) -> tuple[str | None, int]:
    if len(parts) >= 2:
        candidate = f"{parts[0]} {parts[1]}".lower()
        if candidate in BIBLE_BOOKS:
            return candidate.title(), 2
    if parts[0].lower() in BIBLE_BOOKS:
        return parts[0].title(), 1
    return None, 0

def parse_reference_from_match(raw_match: str) -> dict | None:
    parts = re.split(r'[\s:]+', raw_match.strip())
    parts = [p for p in parts if p]

    book, after_book_idx = find_book_in_parts(parts)
    if not book:
        return None

    remaining = parts[after_book_idx:]
    chapter = remaining[0] if len(remaining) > 0 else None
    verse   = remaining[1] if len(remaining) > 1 else None

    if not chapter or not chapter.isdigit():
        return None

    return {
        "full": raw_match.strip().title(),
        "book": book,
        "chapter": chapter,
        "verse": verse,
    }

def extract_bible_reference(text: str) -> list[dict]:
    text = normalize_text(text)

    pattern = (
        r'\b'
        r'(?:[1-4]\s+)?'
        r'[A-Za-z]+(?:\s+[A-Za-z]+)?'
        r'\s+'
        r'\d+'
        r'(?:[\s:]\d+)?'
        r'\b'
    )

    results = []
    seen = set()

    for match in re.finditer(pattern, text, re.IGNORECASE):
        raw = match.group()
        parsed = parse_reference_from_match(raw)
        if parsed is None:
            continue

        key = (parsed["book"], parsed["chapter"], parsed["verse"])
        if key in seen:
            continue
        seen.add(key)
        results.append(parsed)

    return results