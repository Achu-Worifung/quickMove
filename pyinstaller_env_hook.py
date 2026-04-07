
import os, sys
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
if hasattr(sys, "_MEIPASS"):
    os.environ["PATH"] = sys._MEIPASS + ";" + os.environ.get("PATH", "")
