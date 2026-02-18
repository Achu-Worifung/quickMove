
import os, sys
if hasattr(sys,"_MEIPASS"):
    os.environ["PATH"]=sys._MEIPASS+";"+os.environ.get("PATH","")
