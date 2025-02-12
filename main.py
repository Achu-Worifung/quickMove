import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidgetItem, QMainWindow
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt5.uic import loadUi
from util import event_tracker, Simulate, Message as msg, dataMod
import util.walk as walk
from util.clearLayout import clearLayout
from widgets.SearchWidget import SearchWidget
from functools import partial

# Setting Application ID (for Windows taskbar icon)
try:
    from ctypes import windll
    myappid = 'amc.quickmove.automatasimulator'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

basedir = os.path.dirname(__file__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI file
        ui_path = os.path.join(os.path.dirname(__file__), './ui/MainWindow.ui')
        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"
        loadUi(ui_path, self)

        # Set app icon and remove window controls
        self.setWindowIcon(QIcon(os.path.join(basedir, "logo.ico")))
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Initialize settings
        self.settings = QSettings("MyApp", "AutomataSimulator")
        self.restore_previous_geometry()

        # Store base directory
        self.settings.setValue("basedir", basedir)

        # Initialize event tracker thread
        self.event_tracker_thread = None

        # Connect button actions
        self.all_buttons_function()
        
        #keeping track of our current page
        self.curr_page = "home"
        self.home = self.created_autos
        
       

        # Load automata list
        self.MainPage()

    def restore_previous_geometry(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("prev_verse", "")
        self.settings.setValue("next_verse", "")
        self.settings.setValue('copied_reference', '')
        event.accept()

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = event.globalPos() - self.oldPos
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()
        
    def toggleMenu(self):
        width = self.functions.width()
        if (width == 0) :
            newWidth = 175
        else:
            newWidth = 0
        self.functions.setFixedWidth(newWidth)
    def moveToCreate(self):
 
        self.toggleMenu()
        self.stackedWidget.setCurrentIndex(1)
        self.curr_page = "create"
        
        from widgets.Create import Create 
        page = self.stackedWidget.layout().itemAt(1).widget()
        self.modepage = Create(page)
        self.stackedWidget.setCurrentIndex(1)
        pass
    def moveToAbout(self):
        #hiding the nav bar
        self.toggleMenu()
        self.stackedWidget.setCurrentIndex(3)
        self.curr_page = "about"
        
        pass
    def moveToHistory(self):
        self.curr_page = "history"
        pass
    def moveTOSearchArea(self):
        self.curr_page = "searchArea"
        pass
    def moveHome(self):
        if self.curr_page == "home":
           self.toggleMenu()
          
        elif self.curr_page == 'create':
            #retrieving the latest data
            self.toggleMenu()
            layout = self.created_autos.layout()
            self.clearLayout(layout)
            self.start(comming_back=True)
            
        else:
            self.toggleMenu()
            self.stackedWidget.setCurrentIndex(0)
            
        self.curr_page = "home"
    def clearLayout(self, layout):
        while layout.count():  # Loop through all items
            child = layout.takeAt(0)  # Remove item at index 0
            if child.widget():
                child.widget().deleteLater()  # Delete the widget
                child.widget().setParent(None)

    def clearWidgets(self):
         # Deleting the widget at index 1 in the layout
        deleted_pane_item = self.mainContent.layout().takeAt(1)

        # Check if the item at index 1 exists
        if deleted_pane_item:
            widget_to_delete = deleted_pane_item.widget()  # Get the widget from the item
            if widget_to_delete:
                # Remove and delete the widget
                self.mainContent.layout().removeWidget(widget_to_delete)
                widget_to_delete.deleteLater()  # Optionally delete the widget if it's no longer needed
                widget_to_delete.setParent(None)
                del widget_to_delete

    def MainPage(self):
        self.stackedWidget.setCurrentIndex(0)
        from widgets.MainPage import MainPage
        page = self.stackedWidget.layout().itemAt(0).widget()
        self.modepage = MainPage(page)
        pass

    

    def configure_search_area(self):
        from util.event_tracker import EventTracker
        msg.informationBox(self, 'Search Area', "Click Ok to get started. Click on the top-left corner, then the bottom-right corner to set the area. Press 'ESC' to stop.")

        def setup_search_area(event):
            if len(event) >= 2:
                left, top = event[0]['location']
                right, bottom = event[1]['location']
                self.settings.setValue("search_area", (left, top, right, bottom))
                msg.informationBox(self, 'Search Area', "Search area configured successfully")
            else:
                msg.warningBox(self, 'Error', 'Please select the search area')

        self.event_tracker_thread = EventTracker()
        self.event_tracker_thread.tracking_finished.connect(setup_search_area)
        self.event_tracker_thread.create_new_automaton()

    def all_buttons_function(self):
        self.close_button.clicked.connect(self.close)
        self.minimize_button.clicked.connect(self.showMinimized)
        self.menu.clicked.connect(self.toggleMenu)
        self.about.clicked.connect(self.moveToAbout)
        self.history.clicked.connect(self.moveToHistory)
        self.searchArea.clicked.connect(self.moveTOSearchArea)
        self.create.clicked.connect(self.moveToCreate)
        self.home.clicked.connect(self.moveHome)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
