
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from util import Message as msg
from PyQt5.QtCore import QSettings
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
        
       
        
        self.show_button.clicked.connect(self.show_search)
        self.configure.clicked.connect(self.configureSearch)
        
        self.updateLable()
        
    def updateLable(self):
        current = self.settings.value('search_area')
        if current:
            self.label.setText(f"Current search area: {current}")
        else:
            self.label.setText("No search area configured")
        
    def show_search(self):
        print('showing search')
        pass
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