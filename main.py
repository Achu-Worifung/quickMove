
import sys
from PyQt5.QtWidgets import QDialog

import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTableWidgetItem, QMainWindow,QSizeGrip
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt5.uic import loadUi
from util import event_tracker, Simulate, Message as msg, dataMod
import util.walk as walk
from util.clearLayout import clearLayout
from widgets.SearchWidget import SearchWidget
from functools import partial
import torch


# Setting Application ID (for Windows taskbar icon)
try:
    from ctypes import windll
    myappid = 'amc.quickmove.automatasimulator'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass


basedir = os.path.dirname(__file__)

#for custom size grip
resize = False
class CustomSizeGrip(QSizeGrip):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Make the size grip larger
        self.setMinimumSize(20, 20)
        self.setCursor(Qt.SizeAllCursor)
        # Set a stylesheet to make it visible
        self.setStyleSheet("""
                QSizeGrip {
                    background-color: #e0e0e0;
                    border: 1px solid #b0b0b0;
                    border-radius: 4px;
                }
                QSizeGrip:hover {
                    background-color: #c0c0c0;
                }
            """)
    def mousePressEvent(self, event):
        # print("Size grip pressed")
        global resize 
        resize = True
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        # print("Size grip released")
        global resize 
        resize = False
        super().mouseReleaseEvent(event)
        
    def mouseMoveEvent(self, event):
        # print("Size grip being dragged")
        super().mouseMoveEvent(event)
        
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
        

         # Replace the regular QSizeGrip with CustomSizeGrip
        sizegrip = CustomSizeGrip(self)
        sizegrip.setVisible(True)
        sizegrip.setObjectName("size_grip")
        sizegrip.setCursor(Qt.SizeAllCursor)
        self.mainContent.layout().addWidget(sizegrip, 0, Qt.AlignBottom | Qt.AlignRight)

        # Initialize settings
        self.settings = QSettings("MyApp", "AutomataSimulator")
        self.restore_previous_geometry()
        
        if self.settings.value("default_settings") is None:
            self.setUpDefaultSettings()
        # Store base directory
        self.settings.setValue("basedir", basedir)

        # Initialize event tracker thread
        self.event_tracker_thread = None

        # Connect button actions
        self.all_buttons_function()
        
        #keeping track of our current page
        self.curr_page = "home"
        self.home = self.created_autos
        
        self.oldPos = None
        
       
        # Load automata list
        self.MainPage()

    def restore_previous_geometry(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        if self.curr_page == "settings":
            self.moveFromSettings()

            
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("prev_verse", "")
        self.settings.setValue("next_verse", "")
        self.settings.setValue('copied_reference', '')
        event.accept()

    def mousePressEvent(self, event):
        """Store the initial position when mouse is pressed"""
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()
        else:
            self.oldPos = None
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Reset position tracking when mouse is released"""
        self.oldPos = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Handle window dragging"""
        global resize
        # Skip movement if we're resizing or if no button was pressed initially
        if resize or self.oldPos is None:
            return
            
        # Calculate how far the mouse has moved since the initial click
        delta = event.globalPos() - self.oldPos
        # Move the window by that amount
        self.move(self.x() + delta.x(), self.y() + delta.y())
        # Update the old position for the next movement
        self.oldPos = event.globalPos()
        
    def toggleMenu(self):
        width = self.functions.width()
        if (width == 0) :
            newWidth = 175
        else:
            newWidth = 0
        self.functions.setFixedWidth(newWidth)
    def moveToCreate(self):
        self.saveSettings()

        self.toggleMenu()
        self.stackedWidget.setCurrentIndex(1)
        self.curr_page = "create"
        
        from widgets.Create import Create 
        page = self.stackedWidget.layout().itemAt(1).widget()
        self.modepage = Create(page)
        self.stackedWidget.setCurrentIndex(1)
        pass
    def moveToAbout(self):
        self.saveSettings()

        #hiding the nav bar
        self.toggleMenu()
        self.stackedWidget.setCurrentIndex(3)
        self.curr_page = "about"
        
        pass
    # def moveToHistory(self):
    #     self.curr_page = "history"
    #     pass
    def saveSettings(self):
        if self.curr_page == "settings":
            self.moveFromSettings()
       
    def moveTOSearchArea(self):
        self.saveSettings()
        self.curr_page = "searchArea"
        self.toggleMenu()
        self.stackedWidget.setCurrentIndex(5)
        from widgets.SearchArea import SearchArea
        page = self.stackedWidget.layout().itemAt(5).widget()
        self.modepage = SearchArea(page)
        pass
    def moveToSettings(self):
        self.saveSettings()

        self.curr_page = "settings"
        self.toggleMenu()
        self.stackedWidget.setCurrentIndex(6)
        self.curr_page = "settings"
        
        from widgets.Settings import Settings 
        page = self.stackedWidget.layout().itemAt(6).widget()
        self.modepage = Settings(page)
        pass
    def moveFromSettings(self):
        # print('were changes made', self.settings.value("changesmade"))
        changes = len(self.modepage.made_changes) == 0 
        print('this is changes', changes)
        if not changes:
            print('were changes made', self.settings.value("changesmade"))
            link = os.path.join(os.path.dirname(__file__), './ui/settingwarning.ui')
            from widgets.Warning import Warnings
            warning_dialog = Warnings(link)
            
            result = warning_dialog.exec_()  # Modal — blocks until user acts

            if result == QDialog.Accepted:
                if warning_dialog.clicked_button == "discard":
                    return
                elif warning_dialog.clicked_button == "save":
                   print('here is the settings page', self.modepage)
                   self.modepage.save_settings()
            self.settings.setValue("changesmade", False)
            self.settings.sync()

 
    def moveHome(self):
        # if self.curr_page == "home":
        #    self.toggleMenu()
        self.saveSettings()

        if self.curr_page == 'create':
            #retrieving the latest data
            self.toggleMenu()
            self.stackedWidget.setCurrentIndex(0)
            # for 
            from widgets.MainPage import MainPage
            page = self.stackedWidget.layout().itemAt(0).widget()
            self.modepage = MainPage(page)
            
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

    def clearWidgets(self, layout):
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

    
    def setUpDefaultSettings(self):
        deafult_processing = "GPU" if torch.cuda.is_available() else "CPU"
        default_cores = max(1, torch.get_num_threads())
        if torch.cuda.is_available():
            self.settings.setValue('default_prcessing', "GPU")
        #keep this order for the settings
        self.settings.setValue("default_settings", [deafult_processing, "Tiny", 1, 0.00, default_cores, 1, 0.90, 1, 5, 1, 1024, 16000, 500])
        
        self.setupInitialSettings()
    
    def setupInitialSettings(self):
        self.settings.setValue('prcessing', "CPU")
        self.settings.setValue('cores', max(1, torch.get_num_threads()))
        self.settings.setValue('model', "Tiny")
        self.settings.setValue('beam', 1)
        self.settings.setValue("best", 1)
        self.settings.setValue("termperature", 0.00)
        self.settings.setValue("language", 'en')
        # self.settings.setValue("vad_filter", True)
        # self.settings.setValue("vad_parameters", True)
        self.settings.setValue("channel", 1)
        self.settings.setValue("rate", 16000)
        self.settings.setValue("chunks", 1024)
        self.settings.setValue("silence", 0.90) #not in settings
        self.settings.setValue("silencelen", 500) #not in settings
        self.settings.setValue("energy", 0.001)
        
        self.settings.setValue("minlen", 1)
        self.settings.setValue("maxlen", 5)
        self.settings.setValue('auto_length', 1)
        self.settings.setValue('suggestion_length', 3)
        # self.settings.setValue("silencelen", 0.90)
        
        

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
        # self.expand.clicked.connect(self.expandWindow)
        self.menu.clicked.connect(self.toggleMenu)
        # self.about.clicked.connect(self.moveToAbout)
        # self.history.clicked.connect(self.moveToHistory)
        self.searchArea.clicked.connect(self.moveTOSearchArea)
        self.create.clicked.connect(self.moveToCreate)
        self.home.clicked.connect(self.moveHome)
        self.settings_btn.clicked.connect(self.moveToSettings)

    # def expandWindow(self):
    #     if self.isMaximized():
    #         self.showNormal()
    #     else:
    #         self.showMaximized()
def main():
    app = QApplication(sys.argv)
    settings = QSettings("MyApp", "AutomataSimulator")
    
    
    window = MainWindow()
    settings.setValue("main_window", window)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()



