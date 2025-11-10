import re

# A (partial) map. You'll need to expand this for all 66 books.
# The KEY is the variation (lowercase), the VALUE is the canonical name in your JSON.
BOOK_MAP = {
    # --- Old Testament ---
    
    # Genesis
    'genesis': 'Genesis', 'gen': 'Genesis', 'ge': 'Genesis', 'gn': 'Genesis',
    
    # Exodus
    'exodus': 'Exodus', 'exo': 'Exodus', 'ex': 'Exodus', 'exod': 'Exodus',
    
    # Leviticus
    'leviticus': 'Leviticus', 'lev': 'Leviticus', 'le': 'Leviticus', 'lv': 'Leviticus',
    
    # Numbers
    'numbers': 'Numbers', 'num': 'Numbers', 'nu': 'Numbers', 'nm': 'Numbers', 'nb': 'Numbers',
    
    # Deuteronomy
    'deuteronomy': 'Deuteronomy', 'deut': 'Deuteronomy', 'de': 'Deuteronomy', 'dt': 'Deuteronomy',
    
    # Joshua
    'joshua': 'Joshua', 'josh': 'Joshua', 'jos': 'Joshua', 'jsh': 'Joshua',
    
    # Judges
    'judges': 'Judges', 'judg': 'Judges', 'jdg': 'Judges', 'jg': 'Judges', 'jdgs': 'Judges',
    
    # Ruth
    'ruth': 'Ruth', 'rth': 'Ruth', 'ru': 'Ruth',
    
    # 1 Samuel
    '1 samuel': '1 Samuel', '1 sam': '1 Samuel', '1 sa': '1 Samuel', '1sm': '1 Samuel', '1sa': '1 Samuel', 'i samuel': '1 Samuel', '1st samuel': '1 Samuel',
    
    # 2 Samuel
    '2 samuel': '2 Samuel', '2 sam': '2 Samuel', '2 sa': '2 Samuel', '2sm': '2 Samuel', '2sa': '2 Samuel', 'ii samuel': '2 Samuel', '2nd samuel': '2 Samuel',
    
    # 1 Kings
    '1 kings': '1 Kings', '1 kgs': '1 Kings', '1 ki': '1 Kings', '1kin': '1 Kings', '1kgs': '1 Kings', 'i kings': '1 Kings', '1st kings': '1 Kings',
    
    # 2 Kings
    '2 kings': '2 Kings', '2 kgs': '2 Kings', '2 ki': '2 Kings', '2kin': '2 Kings', '2kgs': '2 Kings', 'ii kings': '2 Kings', '2nd kings': '2 Kings',
    
    # 1 Chronicles
    '1 chronicles': '1 Chronicles', '1 chron': '1 Chronicles', '1 ch': '1 Chronicles', '1chr': '1 Chronicles', 'i chronicles': '1 Chronicles', '1st chronicles': '1 Chronicles',
    
    # 2 Chronicles
    '2 chronicles': '2 Chronicles', '2 chron': '2 Chronicles', '2 ch': '2 Chronicles', '2chr': '2 Chronicles', 'ii chronicles': '2 Chronicles', '2nd chronicles': '2 Chronicles',
    
    # Ezra
    'ezra': 'Ezra', 'ezr': 'Ezra', 'ez': 'Ezra',
    
    # Nehemiah
    'nehemiah': 'Nehemiah', 'neh': 'Nehemiah', 'ne': 'Nehemiah',
    
    # Esther
    'esther': 'Esther', 'esth': 'Esther', 'es': 'Esther',
    
    # Job
    'job': 'Job', 'jb': 'Job',
    
    # Psalms
    'psalms': 'Psalms', 'psalm': 'Psalms', 'ps': 'Psalms', 'psa': 'Psalms',
    
    # Proverbs
    'proverbs': 'Proverbs', 'prov': 'Proverbs', 'pro': 'Proverbs', 'pr': 'Proverbs', 'prv': 'Proverbs',
    
    # Ecclesiastes
    'ecclesiastes': 'Ecclesiastes', 'eccles': 'Ecclesiastes', 'ecc': 'Ecclesiastes', 'ec': 'Ecclesiastes', 'qoh': 'Ecclesiastes',
    
    # Song of Songs
    'song of songs': 'Song of Songs', 'song of solomon': 'Song of Songs', 'sos': 'Song of Songs', 'so': 'Song of Songs', 'canticles': 'Song of Songs', 'cant': 'Song of Songs',
    
    # Isaiah
    'isaiah': 'Isaiah', 'isa': 'Isaiah', 'is': 'Isaiah',
    
    # Jeremiah
    'jeremiah': 'Jeremiah', 'jer': 'Jeremiah', 'je': 'Jeremiah', 'jr': 'Jeremiah',
    
    # Lamentations
    'lamentations': 'Lamentations', 'lam': 'Lamentations', 'la': 'Lamentations',
    
    # Ezekiel
    'ezekiel': 'Ezekiel', 'ezek': 'Ezekiel', 'eze': 'Ezekiel', 'ezk': 'Ezekiel',
    
    # Daniel
    'daniel': 'Daniel', 'dan': 'Daniel', 'da': 'Daniel', 'dn': 'Daniel',
    
    # Hosea
    'hosea': 'Hosea', 'hos': 'Hosea', 'ho': 'Hosea',
    
    # Joel
    'joel': 'Joel', 'jl': 'Joel',
    
    # Amos
    'amos': 'Amos', 'am': 'Amos',
    
    # Obadiah
    'obadiah': 'Obadiah', 'obad': 'Obadiah', 'ob': 'Obadiah', 'oba': 'Obadiah',
    
    # Jonah
    'jonah': 'Jonah', 'jon': 'Jonah', 'jnh': 'Jonah',
    
    # Micah
    'micah': 'Micah', 'mic': 'Micah', 'mi': 'Micah',
    
    # Nahum
    'nahum': 'Nahum', 'nah': 'Nahum', 'na': 'Nahum',
    
    # Habakkuk
    'habakkuk': 'Habakkuk', 'hab': 'Habakkuk', 'ha': 'Habakkuk', 'hb': 'Habakkuk',
    
    # Zephaniah
    'zephaniah': 'Zephaniah', 'zeph': 'Zephaniah', 'zep': 'Zephaniah', 'zp': 'Zephaniah',
    
    # Haggai
    'haggai': 'Haggai', 'hag': 'Haggai', 'hg': 'Haggai',
    
    # Zechariah
    'zechariah': 'Zechariah', 'zech': 'Zechariah', 'zec': 'Zechariah', 'zk': 'Zechariah',
    
    # Malachi
    'malachi': 'Malachi', 'mal': 'Malachi', 'ml': 'Malachi',

    # --- New Testament ---
    
    # Matthew
    'matthew': 'Matthew', 'matt': 'Matthew', 'mt': 'Matthew',
    
    # Mark
    'mark': 'Mark', 'mrk': 'Mark', 'mar': 'Mark', 'mk': 'Mark', 'mr': 'Mark',
    
    # Luke
    'luke': 'Luke', 'luk': 'Luke', 'lk': 'Luke',
    
    # John
    'john': 'John', 'joh': 'John', 'jn': 'John', 'jhn': 'John',
    
    # Acts
    'acts': 'Acts', 'act': 'Acts', 'ac': 'Acts',
    
    # Romans
    'romans': 'Romans', 'rom': 'Romans', 'ro': 'Romans', 'rm': 'Romans',
    
    # 1 Corinthians
    '1 corinthians': '1 Corinthians', '1 cor': '1 Corinthians', '1 co': '1 Corinthians', '1cor': '1 Corinthians', 'i corinthians': '1 Corinthians', '1st corinthians': '1 Corinthians',
    
    # 2 Corinthians
    '2 corinthians': '2 Corinthians', '2 cor': '2 Corinthians', '2 co': '2 Corinthians', '2cor': '2 Corinthians', 'ii corinthians': '2 Corinthians', '2nd corinthians': '2 Corinthians',
    
    # Galatians
    'galatians': 'Galatians', 'gal': 'Galatians', 'ga': 'Galatians',
    
    # Ephesians
    'ephesians': 'Ephesians', 'eph': 'Ephesians', 'ep': 'Ephesians',
    
    # Philippians
    'philippians': 'Philippians', 'phil': 'Philippians', 'php': 'Philippians', 'pp': 'Philippians',
    
    # Colossians
    'colossians': 'Colossians', 'col': 'Colossians', 'co': 'Colossians',
    
    # 1 Thessalonians
    '1 thessalonians': '1 Thessalonians', '1 thess': '1 Thessalonians', '1 thes': '1 Thessalonians', '1 th': '1 Thessalonians', '1thess': '1 Thessalonians', 'i thessalonians': '1 Thessalonians', '1st thessalonians': '1 Thessalonians',
    
    # 2 Thessalonians
    '2 thessalonians': '2 Thessalonians', '2 thess': '2 Thessalonians', '2 thes': '2 Thessalonians', '2 th': '2 Thessalonians', '2thess': '2 Thessalonians', 'ii thessalonians': '2 Thessalonians', '2nd thessalonians': '2 Thessalonians',
    
    # 1 Timothy
    '1 timothy': '1 Timothy', '1 tim': '1 Timothy', '1 ti': '1 Timothy', '1tim': '1 Timothy', 'i timothy': '1 Timothy', '1st timothy': '1 Timothy',
    
    # 2 Timothy
    '2 timothy': '2 Timothy', '2 tim': '2 Timothy', '2 ti': '2 Timothy', '2tim': '2 Timothy', 'ii timothy': '2 Timothy', '2nd timothy': '2 Timothy',
    
    # Titus
    'titus': 'Titus', 'tit': 'Titus', 'ti': 'Titus',
    
    # Philemon
    'philemon': 'Philemon', 'philem': 'Philemon', 'phm': 'Philemon', 'pm': 'Philemon',
    
    # Hebrews
    'hebrews': 'Hebrews', 'heb': 'Hebrews', 'he': 'Hebrews',
    
    # James
    'james': 'James', 'jas': 'James', 'jm': 'James',
    
    # 1 Peter
    '1 peter': '1 Peter', '1 pet': '1 Peter', '1 pe': '1 Peter', '1pt': '1 Peter', '1pet': '1 Peter', 'i peter': '1 Peter', '1st peter': '1 Peter',
    
    # 2 Peter
    '2 peter': '2 Peter', '2 pet': '2 Peter', '2 pe': '2 Peter', '2pt': '2 Peter', '2pet': '2 Peter', 'ii peter': '2 Peter', '2nd peter': '2 Peter',
    
    # 1 John
    '1 john': '1 John', '1 jn': '1 John', '1joh': '1 John', '1john': '1 John', 'i john': '1 John', '1st john': '1 John',
    
    # 2 John
    '2 john': '2 John', '2 jn': '2 John', '2joh': '2 John', '2john': '2 John', 'ii john': '2 John', '2nd john': '2 John',
    
    # 3 John
    '3 john': '3 John', '3 jn': '3 John', '3joh': '3 John', '3john': '3 John', 'iii john': '3 John', '3rd john': '3 John',
    
    # Jude
    'jude': 'Jude', 'jud': 'Jude', 'jd': 'Jude',
    
    # Revelation
    'revelation': 'Revelation', 'rev': 'Revelation', 're': 'Revelation', 'the apocalypse': 'Revelation',
}

def normalize_book_name(book_str):
    """
    Converts a parsed book name into its canonical form.
    """
    clean_book = book_str.lower().strip().replace('.', '')
    
    
    match = re.match(r'^([1-3i]{1,3})(\s+)(.*)', clean_book)
    
    if match:
        num = match.group(1)
        book_name = match.group(3)
        book_name = book_name.capitalize().strip()
        # Standardize number
        if num == '1' or num == 'i': num = '1'
        elif num == '2' or num == 'ii': num = '2'
        elif num == '3' or num == 'iii': num = '3'
        
      
        normalized_attempt = f"{num} {book_name}"
        if normalized_attempt in BOOK_MAP:
            return BOOK_MAP[normalized_attempt]
            
     
        normalized_attempt_no_space = f"{num}{book_name}"
        if normalized_attempt_no_space in BOOK_MAP:
            return BOOK_MAP[normalized_attempt_no_space]
            
    
    if clean_book in BOOK_MAP:
        return BOOK_MAP[clean_book]
        

    print(f"Warning: Book name '{book_str}' not found in normalization map.")
    return book_str.strip() 

def getReference(text):
    """
    Finds the first parsable Bible reference in a block of text.
    """
    pattern = r'\b([1-3I]?\s*[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d{1,3}:\d{1,3})\b'

    matches = re.findall(pattern, text)
    
    for match in matches:
        book_part, reference_part = match
       
        book, chapter, verse = parseReference(f"{book_part} {reference_part}")
        if book and chapter and verse:
           
            return f"{book} {chapter}:{verse}"
            
    return None
def parseReference(reference):
    """
    Parses a reference string and returns normalized book, chapter, and verse.
    """
    match = re.match(
        r'^\s*(?P<book>((?:[1-3I]?\s?)?[A-Za-z]+(?:\s+[A-Za-z]+)?))\s+(?P<chapter>\d+):(?P<verse>\d+)\s*$', 
        reference
    )
    
    if match:
        book_str = match.group('book').strip()
        chapter = int(match.group('chapter'))
        verse = int(match.group('verse'))
        

        normalized_book = normalize_book_name(book_str)
        
        if normalized_book:
            return normalized_book, chapter, verse
            
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