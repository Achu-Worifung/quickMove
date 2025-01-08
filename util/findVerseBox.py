import pyautogui
from PyQt5.QtCore import QSettings
import pytesseract
import cv2
import numpy as np

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def findPrevDisplayedVerse():
    try:
        settings = QSettings("MyApp", "AutomataSimulator")
        img_area = settings.value('search_area', None)

        if img_area and isinstance(img_area, (list, tuple)):  # Ensure it's a tuple or list
            # Convert list to tuple if needed
            img_area = tuple(img_area)
            print('Image area:', img_area)
            
            # Take a screenshot of the specified region using pyautogui
            screenshot = pyautogui.screenshot(region=img_area)
            
            # Convert the screenshot to OpenCV format
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

            # Preprocessing for better OCR accuracy
            gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # Optional: Save preprocessed image
            cv2.imwrite("preprocessed_image.png", thresh)
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh)
            print("Extracted Text:", text)
            return text
        else:
            print("Search area not defined or invalid format.")
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None

# Example usage
if __name__ == "__main__":
    findPrevDisplayedVerse()
