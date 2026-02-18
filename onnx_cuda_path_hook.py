
import os, sys
if hasattr(sys, "_MEIPASS"):
    os.environ["PATH"] = sys._MEIPASS + ";" + os.environ.get("PATH","")
    os.environ["ORT_TENSORRT_UNAVAILABLE"]="1"
    meipass = sys._MEIPASS
    if os.path.exists(meipass) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(meipass)
