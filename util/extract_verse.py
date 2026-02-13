import re 

def extract_bible_reference(text):
    word_pattern =     text = re.sub(r'chapter\s+(\d+)\s+verse\s+(\d+)', r'\1:\2', text)
    pattern = r'\b(?:[1-4]\s+)?[a-z]+\s+\d+:\d+(?:-\d+)?\b'
    
    matches = re.finditer(pattern, word_pattern)
    
    results = []
    
    for match in matches:
        raw_match = match.group()
        parts = re.split(r'\s+|:', raw_match)
        
        # Determine if it's a numbered book (e.g., 2 Kings) or single (e.g., Genesis)
        if parts[0].isdigit():
            book = f"{parts[0]} {parts[1]}"
            chapter = parts[2]
            verse = parts[3]
        else:
            book = parts[0]
            chapter = parts[1]
            verse = parts[2]
            
        results.append({
            "full": raw_match.title(), 
            "book": book.title(),
            "chapter": chapter,
            "verse": verse
        })
        
    return results
    
