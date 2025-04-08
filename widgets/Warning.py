from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

class Warnings(QDialog):
    def __init__(self, link=None):
        super().__init__()
        self.link = link
        loadUi(link, self) 
        self.setWindowTitle("Warning")
        self.show() 
        self.clicked_button = None
        self.discardchanges.clicked.connect(self.discard_changes)
        self.savechanges.clicked.connect(self.save_changes)
    
    def discard_changes(self):
        self.clicked_button = "discard"
        self.accept()
    def save_changes(self):
        self.clicked_button = "save"
        self.accept()