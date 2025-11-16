from PyQt5.QtWidgets import QMainWindow, QPushButton
from PyQt5 import QtWidgets
from pyqttoast import Toast, ToastPreset


class Window(QMainWindow):
    def __init__(self):
        super().__init__(parent=None)

        # Add button and connect click event
        self.button = QPushButton(self)
        self.button.setText('Show toast')
        self.button.clicked.connect(self.show_toast)
    
    # Shows a toast notification every time the button is clicked
    def show_toast(self):
        toast = Toast()
        toast.setDuration(0)  # Hide after 5 seconds
        toast.setTitle('Success! Confirmation email sent.')
        toast.setText('Check your email to complete signup.')
        toast.applyPreset(ToastPreset.INFORMATION)  # Apply style preset
        toast.show()
        toast = Toast()
        toast.setDuration(0)  # Hide after 5 seconds
        toast.setTitle('Success! SUCCESS.')
        toast.setText('Check your email to complete signup.')
        toast.applyPreset(ToastPreset.SUCCESS)  # Apply style preset
        toast.show()

app = QtWidgets.QApplication([])
window = Window()
window.show()
app.exec_()