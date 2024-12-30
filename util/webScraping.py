import requests
from bs4 import BeautifulSoup
import re


def get_verse(verse, version="NIV"):
    url = f"https://www.biblegateway.com/passage/?search={verse}&version={version}"

    # Making the request
    r = requests.get(url)

    # Parsing the HTML
    soup = BeautifulSoup(r.content, "html.parser")

    # Locate the desired section containing the verse
    desired_section = soup.find_all('div', class_="std-text")
    
    if desired_section:
        # Extract text from the desired section and remove occurrences like "(A)"
        raw_text = desired_section[0].text
        cleaned_text = re.sub(r"\([A-Z]\)", '', raw_text)  # Remove patterns like "(A)"
        cleaned_text = re.sub(r"\s{2,}", ' ', cleaned_text).strip()  # Clean up extra spaces

        return cleaned_text
    



