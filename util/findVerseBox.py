"""this module will use pyautogui to find the location of the specified verse box in propresenter 7 using the screenshot locate function."""

import pyautogui
from PyQt5.QtCore import QSettings #for the search area
import os


def findVerseBox_location():
    try:
        settings = QSettings("MyApp", "AutomataSimulator")
        basedir = settings.value("basedir", None)
        image_location = os.path.join(basedir, 'images', 'test2.png')
        search_area = settings.value("search_area", None)  # Use None as default if not set

        x,y = pyautogui.locateCenterOnScreen(image_location,confidence=0.9, region=search_area)

        if x and y:
            print("Found the home button at:", x, y)
            # pyautogui.moveTo(x, y, duration=1)
        else:
            print("Could not find the home button")
        pass
    #return the x and y coordinates
        return x,y
    except Exception as e:
        print("An error occurred while finding the verse box location:", e)
        return None, None