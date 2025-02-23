import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QBrush, QColor
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow, QLabel
from PyQt5.QtWidgets import *

class HighlightWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Highlight Square Example")
        self.setGeometry(100, 100, 300, 200)

        # Add a button to trigger the square highlight
        self.highlight_button = QPushButton("Highlight Square", self)
        self.highlight_button.setGeometry(50, 50, 200, 50)
        self.highlight_button.clicked.connect(self.show_highlight)

        # Create an overlay for the highlight effect
        self.overlay = HighlightOverlay()

    def show_highlight(self):
        # Display the overlay to highlight a square
        self.overlay.show()


class HighlightOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Get the geometry of all connected screens
        screen_geometry = QApplication.desktop().screenGeometry()
        self.setGeometry(screen_geometry)

        self.square_rect = QRect(1900, 900, 200, 200)  # Example square position and size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))  # Semi-transparent black background
        painter.drawRect(self.rect())

        # Draw the highlighted square
        painter.setBrush(QBrush(QColor(255, 255, 0, 150)))  # Semi-transparent yellow
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.square_rect)

    def mousePressEvent(self, event):
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HighlightWindow()
    window.show()
    sys.exit(app.exec_())
