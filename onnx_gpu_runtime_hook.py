import os
import sys

# ensure CUDA path is visible
os.environ["PATH"] = os.environ.get("PATH", "") + ";" + os.path.join(sys._MEIPASS, "")
