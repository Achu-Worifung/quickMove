import re

def getReference(text):
    # Regex pattern to capture Bible references
    pattern = r'\b(?:[1-3]?\s?[A-Za-z]+)\s+\d+:\d+(?:-\d+)?(?:,\d+)*\b'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches

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