import re

def getReference(text):
    # Regex pattern to capture single Bible references (e.g., Genesis 1:1)
    pattern = r'\b(?:[1-3]?\s?[A-Za-z]+)\s+\d+:\d+\b'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Return the first match or None if no match is found
    return matches[0] if matches else None
def parseReference(reference):
    # Split the reference into book, chapter, and verse
    match = re.match(r'^(?P<book>(?:[1-3]?\s?[A-Za-z]+))\s+(?P<chapter>\d+):(?P<verse>\d+)$', reference)
    if match:
        book = match.group('book').strip()
        chapter = int(match.group('chapter'))
        verse = int(match.group('verse'))
        return book.capitalize().strip(), str(chapter).strip(), str(verse).strip()
    return None, None, None
def boldedText(text, enteredText):

    # print('text', text)
    # print('enteredText', enteredText.split())
    # Get text and words to bold
   
    words = enteredText.split()

    # Escape and create a regex pattern for the words
    escaped_words = [re.escape(word.strip()) for word in words]
    pattern = r'\b(' + '|'.join(escaped_words) + r')\b'

    # Replace matches with bold HTML tags
    bolded_text = re.sub(pattern, r'<b>\1</b>', text, flags=re.IGNORECASE)

    return bolded_text