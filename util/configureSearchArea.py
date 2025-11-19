from pynput import  mouse
from PyQt5.QtCore import QObject, pyqtSignal
import ctypes

class SearchArea(QObject):
    tracking_finished = pyqtSignal(tuple)  
    def __init__(self):
        super().__init__()
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        
        self.listener = None
    def configure_search_area(self):
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()
        self.listener.join()
    def on_click(self, x, y, button, pressed):
        if pressed:
            # Get the scaling factor
            scaling_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
            adjusted_x = int(x / scaling_factor)
            adjusted_y = int(y / scaling_factor)

            if self.x1 is None and self.y1 is None:
                self.x1, self.y1 = adjusted_x, adjusted_y
                print(f"First corner set at: ({self.x1}, {self.y1})")
            else:
                self.x2, self.y2 = adjusted_x, adjusted_y
                print(f"Second corner set at: ({self.x2}, {self.y2})")
                # Stop listener
                self.tracking_finished.emit((self.x1, self.y1, self.x2, self.y2))
                self.listener.stop()