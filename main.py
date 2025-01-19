import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow
from widgets.Welcome import Welcome  # Correctly import the Welcome class
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSettings

basedir = os.path.dirname(__file__)

try:
    from ctypes import windll
    myappid = 'amc.quickmove.automatasimulator'  # arbitrary string
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setCentralWidget(Welcome())
        self.setWindowTitle("Automata Simulator")
        self.setWindowIcon(QIcon(os.path.join(basedir,"logo.ico")))  # Ensure icon.png is in the same directory as this script
        self.area = None

        # Initialize QSettings for storing geometry and search area
        self.settings = QSettings("MyApp", "AutomataSimulator")
        self.restore_previous_geometry()
        self.search_area()  # Retrieve the saved search area setting
        
        #reseting the search bar area
        self.settings.setValue("bar_location", None)
        self.settings.setValue("prev_verse", "")
        self.settings.setValue("next_verse", "")
        self.settings.setValue('copied_reference', '')
        
        #setting the basedirectory in the settings
        self.settings.setValue("basedir", basedir)

    def closeEvent(self, event):
        # Save the current geometry when the app is closed
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def restore_previous_geometry(self):
        # Restore geometry if available
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def search_area(self):
        # Load the saved search_area from QSettings
        self.area = self.settings.value("search_area", None)  # Use None as default if not set
        if self.area:
            print(f"Restored search area: {self.area}")
        else:
            print("No search area saved. Using default.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Exiting due to: {e}")


if __name__ == "__main__":
    main()
