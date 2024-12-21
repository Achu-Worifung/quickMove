"""
"""

import sys 
from PyQt5.QtWidgets import QMessageBox, QInputDialog

def informationBox(self,title, text):
    confirmationbox = QMessageBox.information(self, 
                                              title,
                                              text
                                              )
    if confirmationbox == QMessageBox.Ok:
        return True
    else:
        return False

def questionBox(self,title, text):
    
    questionBox = QMessageBox.question(self, 
                                       title,
                                       text,
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
    if questionBox == QMessageBox.Yes:
        return True
    else:
        return False


def warningBox(self, title, text):
    warningBox = QMessageBox.warning(self, 
                                     title,
                                     text)
    return warningBox

def inputDialogBox(self, title, text):
    text, ok = QInputDialog.getText(self, title, text)
    return text, ok