import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QPushButton, QMainWindow,
                           QWidget, QVBoxLayout)
from PyQt5.QtCore import QRect

overlays = []
class HighlightWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.setWindowTitle("Screen Overlay")
        self.setGeometry(100, 100, 300, 100)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add a button to trigger the overlay
        self.overlay_button = QPushButton("Show Overlay")
        layout.addWidget(self.overlay_button)
        self.overlay_button.clicked.connect(self.show_overlay)
        
        # Create list to store overlays
        global overlays
        overlays = []
       
    
    def show_overlay(self):
        # Clear any existing overlays
        for overlay in overlays:
            overlay.close()
            overlays.clear()
        
        # Create an overlay for each screen
        for screen in QApplication.screens():
            overlay = ScreenOverlay(screen)
            overlays.append(overlay)
            overlay.show()


class ScreenOverlay(QWidget):
    def __init__(self, screen):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set the geometry to cover the entire screen
        self.setGeometry(screen.geometry())
        
        self.square_rect = QRect(1900, 900, 200, 200)  # Example square position and size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))  # Semi-transparent black
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
         # Draw the highlighted square
        painter.setBrush(QBrush(QColor(255, 255, 0, 150)))  # Semi-transparent yellow
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.square_rect)
    
    def mousePressEvent(self, event):
        global overlays
        for overlay in overlays:
            overlay.close()
            # overlay.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HighlightWindow()
    window.show()
    sys.exit(app.exec_())