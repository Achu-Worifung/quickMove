# ocr_util.py  (refactor of your module)
import importlib
import threading
from typing import Optional
from PyQt5.QtCore import QSettings
from util.util import resource_path
from util import Message as msg
import os
import mss 

# lightweight modules we keep top-level:
# QSettings is cheap so we keep it
# heavy modules will be loaded lazily
_pyautogui = None
_pytesseract = None
_cv2 = None
_np = None

_import_lock = threading.Lock()


def get_numpy():
    global _np
    if _np is None:
        with _import_lock:
            if _np is None:
                _np = importlib.import_module("numpy")
    return _np

def get_cv2():
    global _cv2
    if _cv2 is None:
        with _import_lock:
            if _cv2 is None:
                _cv2 = importlib.import_module("cv2")
    return _cv2

def get_pytesseract():
    """
    Lazily import pytesseract and configure tesseract_cmd and TESSDATA_PREFIX.
    """
    global _pytesseract
    if _pytesseract is None:
        with _import_lock:
            if _pytesseract is None:
                _pytesseract = importlib.import_module("pytesseract")

                # Set Tesseract path (Windows) when pytesseract becomes available
                bundled_tesseract = resource_path("tesseract.exe")
                system_tesseract  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.path.exists(bundled_tesseract):
                    _pytesseract.pytesseract.tesseract_cmd = bundled_tesseract
                    os.environ["TESSDATA_PREFIX"] = resource_path("tessdata")
                elif os.path.exists(system_tesseract):
                    _pytesseract.pytesseract.tesseract_cmd = system_tesseract
                    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
                else:
                    # show message via util.Message (UI)
                    msg.warningBox(None, "Tesseract Not Found",
                                   "Tesseract OCR executable not found. Please install Tesseract OCR "
                                   "from https://github.com/UB-Mannheim/tesseract/wiki")
    return _pytesseract

def findPrevDisplayedVerse() -> Optional[str]:
    """
    Capture region screenshot, preprocess and run OCR.
    Lazy-loads heavy libs on first call.
    """
    try:
        settings = QSettings("MyApp", "AutomataSimulator")
        img_area = settings.value('search_area', None)

        if img_area and isinstance(img_area, (list, tuple)):
            img_area = tuple(img_area)

            np = get_numpy()
            cv2 = get_cv2()
            pytesseract = get_pytesseract()

            # screenshot (PIL Image)
            with mss.mss() as sct:
                 monitor = {
                     "top": img_area[1],
                     "left": img_area[0],
                     "width": img_area[2] - img_area[0],
                     "height": img_area[3] - img_area[1],
                 }
                 sct_img = sct.grab(monitor)

                 screenshot_np = np.array(sct_img)

            # Convert the screenshot to OpenCV format
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

            # Preprocessing for better OCR accuracy
            gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # Perform OCR
            text = pytesseract.image_to_string(thresh)
            text = text.split("(")[0].strip()
            print('here is the box text:', text)
            return text
        else:
            return None
    except Exception as e:
        # Consider logging rather than print in real app
        print("An error occurred in findPrevDisplayedVerse():", e)
        return None

# Optional: background preloader you can call after showing UI
def preload_ocr_modules_async(warm_cuda: bool = False):
    """
    Spawn a daemon thread that imports heavy libs in the background.
    Set warm_cuda=True to do a tiny tensor allocation if torch is used (not used here).
    """
    def _job():
        try:
            get_numpy()
            get_cv2()
            get_pytesseract()
            # optional extra warming steps, e.g., small cv2 operation
        except Exception:
            # ignore failures here; user will see message when using OCR
            pass

    t = threading.Thread(target=_job, daemon=True)
    t.start()

# If this file is run standalone for dev-testing
if __name__ == "__main__":
    print("Testing OCR (lazy) — this will import modules on demand.")
    print(findPrevDisplayedVerse())
