import os
from PyQt5.QtWidgets import (
    QApplication, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QDialog, QLabel, QPushButton,QHBoxLayout, QRadioButton, QPushButton
)
from PyQt5.uic import loadUi

import  util.savedVerses as savedVerses
class search(QDialog):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/intro.ui')
        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"

        loadUi(ui_path, self)

        #adding action event to the version
        version.addItem(['','KJV', 'NIV', 'ESV'])
        version.setCurrentIndex(0)
        # version.currentIndexChanged.connect(self.searchVerse)
        current_text = self.version.currentText()




        



