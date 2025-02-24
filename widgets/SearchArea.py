
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from util import Message as msg
from PyQt5.QtCore import QSettings
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QBrush, QColor
from PyQt5.QtWidgets import (QApplication, QPushButton, QMainWindow,
                           QWidget, QVBoxLayout)
from PyQt5.QtCore import QRect

overlays = []
class SearchArea(QWidget):
    def __init__(self, page_widget):
        super().__init__()
        self.page_widget = page_widget

        self.settings = QSettings("MyApp", "AutomataSimulator")
        self.init_page()
        
    def init_page(self):
        self.show_button = self.page_widget.findChild(QPushButton, 'show_area')
        self.configure= self.page_widget.findChild(QPushButton, 'configure')
        
        self.label = self.page_widget.findChild(QLabel, 'current_config')
        
       
        
        self.show_button.clicked.connect(self.show_overlay)
        self.configure.clicked.connect(self.configureSearch)
        
        self.updateLable()
        
    def updateLable(self):
        current = self.settings.value('search_area')
        if current:
            self.label.setText(f"Current search area: {current}")
        else:
            self.label.setText("No search area configured")
        

    def show_overlay(self):
        global overlays
        overlays = []
        # Clear any existing overlays
        for overlay in overlays:
            overlay.close()
            overlays.clear()
        
        # Create an overlay for each screen
        for screen in QApplication.screens():
            overlay = ScreenOverlay(screen)
            overlays.append(overlay)
            overlay.show()

    def configureSearch(self):
        from util.event_tracker import EventTracker
        msg.informationBox(self, 'Search Area', "Click Ok to get started. Click on the top-left corner, then the bottom-right corner to set the area. Press 'ESC' to stop.")

        def setup_search_area(event):
            if len(event) >= 2:
                left, top = event[0]['location']
                right, bottom = event[1]['location']
                self.settings.setValue("search_area", (left, top, right, bottom))
                self.updateLable()
                msg.informationBox(self, 'Search Area', "Search area configured successfully")
            else:
                msg.warningBox(self, 'Error', 'Please select the search area')

        self.event_tracker_thread = EventTracker()
        self.event_tracker_thread.tracking_finished.connect(setup_search_area)
        self.event_tracker_thread.create_new_automaton()
        
class ScreenOverlay(QWidget):
    def __init__(self, screen):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set the geometry to cover the entire screen
        self.setGeometry(screen.geometry())
        self.geometry = screen.geometry()
        
        self.settings = QSettings("MyApp", "AutomataSimulator")
        
        self.search_area = self.settings.value('search_area')
        
        self.square_rect = QRect(self.search_area[0], self.search_area[1], self.search_area[2] - self.search_area[0], self.search_area[3] - self.search_area[1])  

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))  
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
        
        if(self.geometry.x() <= self.search_area[0] <= self.geometry.x() + self.geometry.width() and self.geometry.y() <= self.search_area[1] <= self.geometry.y() + self.geometry.height()):
            # Draw the highlighted square
            painter.setBrush(QBrush(QColor(255, 255, 255, 100))) 
            painter.setPen(Qt.NoPen)
            painter.drawRect(self.square_rect)
    
    def mousePressEvent(self, event):
        global overlays
        for overlay in overlays:
            overlay.close()
            # overlay.clear()