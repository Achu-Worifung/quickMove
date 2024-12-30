import requests
from bs4 import BeautifulSoup


def get_verse(verse, version= "NIV"):
    url = f"https://www.biblegateway.com/passage/?search={verse}&version={version}"

    #making the request
    r = requests.get(url)

    #parsing the html
    soup = BeautifulSoup(r.content, "html.parser")

    s = soup.find('span', class_='woj')
    content = soup.find_all('#text')

    print(content)

get_verse("John 3:16", 'NIV')