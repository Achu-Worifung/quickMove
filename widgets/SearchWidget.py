import os
from PyQt5.QtWidgets import (
    QApplication, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QDialog, QLabel, QPushButton,QHBoxLayout, QRadioButton, QPushButton
)
from PyQt5.uic import loadUi
from widgets.SearchBar import AutocompleteWidget
import  util.savedVerses as savedVerses
class SearchWidget(QDialog):
    def __init__(self, data = None):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/search.ui')
        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"

        loadUi(ui_path, self)

        #adding action event to the version
        self.version.addItems(['','KJV', 'NIV', 'ESV'])
        self.version.setCurrentIndex(0)
        self.data = data
        print('data in search widget', data)    
        
        #getting the search bar
        autoComplete_widget = AutocompleteWidget(self)
        self.horizontalLayout.addWidget(autoComplete_widget)
        
        # version.currentIndexChanged.connect(self.searchVerse)
        current_text = self.version.currentText()




        



