"""this module will use pyautogui to find the location of the specified verse box in propresenter 7 using the screenshot locate function."""

import pyautogui
from PyQt5.QtCore import QSettings #for the search area

def findVerseBox_location():
    image = './test.png'
    settings = QSettings("MyApp", "AutomataSimulator")
    search_area = settings.value("search_area", None)  # Use None as default if not set

    home_button = pyautogui.locateOnScreen(image, region=search_area)

    if home_button:
        print("Found the home button at:", home_button)
    else:
        print("Could not find the home button")
    pass