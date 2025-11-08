import sys
from PyQt5.QtWidgets import QMessageBox, QInputDialog
from PyQt5.QtGui import QFont

# Common stylesheet for all boxes
MESSAGE_BOX_STYLE = """
QMessageBox {
    background-color: #fdfdfd;
    color: #222;
    font-family: 'Segoe UI';
    font-size: 10.5pt;
}
QLabel {
    color: #222;
    font-size: 10.5pt;
}
QPushButton {
    background-color: #0078d7;
    color: white;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2893f0;
}
QPushButton:pressed {
    background-color: #005a9e;
}
"""

def informationBox(self, title, text):
    box = QMessageBox(self)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(title)
    box.setText(f"ℹ️  <b>{text}</b>")
    box.setStandardButtons(QMessageBox.Ok)
    box.setStyleSheet(MESSAGE_BOX_STYLE)
    result = box.exec_()
    return result == QMessageBox.Ok


def questionBox(self, title, text):
    box = QMessageBox(self)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(title)
    box.setText(f"❓  <b>{text}</b>")
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    box.setDefaultButton(QMessageBox.No)
    box.setStyleSheet(MESSAGE_BOX_STYLE)
    result = box.exec_()
    return result == QMessageBox.Yes


def warningBox(self, title, text):
    box = QMessageBox(self)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(title)
    box.setText(f"⚠️  <b>{text}</b>")
    box.setStandardButtons(QMessageBox.Ok)
    box.setStyleSheet(MESSAGE_BOX_STYLE)
    return box.exec_()


def inputDialogBox(self, title, text):
    dlg = QInputDialog(self)
    dlg.setWindowTitle(title)
    dlg.setLabelText(text)
    dlg.setStyleSheet("""
        QInputDialog {
            background-color: #fdfdfd;
            color: #222;
            font-family: 'Segoe UI';
            font-size: 10.5pt;
        }
        QPushButton {
            background-color: #0078d7;
            color: white;
            border-radius: 6px;
            padding: 6px 16px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #2893f0;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QLineEdit {
            border: 1px solid #aaa;
            border-radius: 4px;
            padding: 4px;
        }
    """)
    text_value, ok = dlg.getText(self, title, text)
    return text_value, ok
