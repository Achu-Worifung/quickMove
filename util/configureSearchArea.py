from pynput import  mouse
from PyQt5.QtCore import QObject, pyqtSignal

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
            if self.x1 is None and self.y1 is None:
                self.x1, self.y1 = x, y
                print(f"First corner set at: ({self.x1}, {self.y1})")
            else:
                self.x2, self.y2 = x, y
                print(f"Second corner set at: ({self.x2}, {self.y2})")
                # Stop listener
                self.tracking_finished.emit((self.x1, self.y1, self.x2, self.y2))
                self.listener.stop()