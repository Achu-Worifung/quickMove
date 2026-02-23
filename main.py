import os
# Force the lazy loader to resolve
os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
os.environ['QT_SCALE_FACTOR'] = '1'
os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'
os.environ["HF_HUB_DISABLE_XET"] = "1"
import transformers
import transformers.modeling_utils
import resources_rc
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import QEvent



from util.modelmanagement import list_downloaded_models
from PyQt5 import QtCore
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
import configparser
import os
import threading

from util.util import resource_path
import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizeGrip, QSplashScreen
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt5.uic import loadUi
from util import  Message as msg
from widgets.SearchWidget import SearchWidget
from functools import partial

from PyQt5 import QtCore, QtGui

# Setting Application ID (for Windows taskbar icon)
if sys.platform == "win32":
    try:
        import ctypes
        
        app_id = u"amc.quickmove.automatasimulator"  
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


basedir = os.path.dirname(__file__)

#for custom size grip

_Settings = None 

def get_settings():
    global _Settings
    if _Settings is None:
        from widgets.Settings import Settings 
        _Settings = Settings
    return _Settings

resize = False
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI file
        ui_path = resource_path('./ui/MainWindow.ui')
        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"
        loadUi(ui_path, self)
        
        #starting the widget resize 
        self._resize_margin = 8  # pixels from edge that trigger resize
        self._resizing = False
        self._resize_direction = None
        self._resize_start_pos = None
        self._resize_start_geometry = None
        self.setMouseTracking(True)
        QApplication.instance().installEventFilter(self)

        # Set app icon and remove window controls
        icon_path = resource_path("logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        self.setWindowFlags(Qt.FramelessWindowHint)
        t = threading.Thread(target=get_settings, daemon=True)
        t.start()


        # Initialize settings
        self.settings = QSettings("MyApp", "AutomataSimulator")
        self.restore_previous_geometry()
        
        # Only run initial setup if it hasn't been done before
        if not self.settings.value("default_settings", False, type=bool):
            print('running initial setup')
            self.setupInitialSettings()
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
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            # map position to main window coordinates
            pos = self.mapFromGlobal(QCursor.pos())
            
            if self._resizing:
                self._do_resize(QCursor.pos())
                return False
            
            direction = self._get_resize_direction(pos)
            if direction:
                self.setCursor(self._get_cursor_for_direction(direction))
            else:
                self.unsetCursor()

        elif event.type() == QEvent.MouseButtonPress:
            pos = self.mapFromGlobal(QCursor.pos())
            if event.button() == Qt.LeftButton:
                direction = self._get_resize_direction(pos)
                if direction:
                    self._resizing = True
                    self._resize_direction = direction
                    self._resize_start_pos = QCursor.pos()
                    self._resize_start_geometry = self.geometry()
                    return True  # consume the event

        elif event.type() == QEvent.MouseButtonRelease:
            if self._resizing:
                self._resizing = False
                self._resize_direction = None
                return True

        return False
    def _get_resize_direction(self, pos):
        """Determine resize direction based on cursor position."""
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        m = self._resize_margin

        left   = x < m
        right  = x > w - m
        top    = y < m
        bottom = y > h - m

        if top and left:     return 'top-left'
        if top and right:    return 'top-right'
        if bottom and left:  return 'bottom-left'
        if bottom and right: return 'bottom-right'
        if left:             return 'left'
        if right:            return 'right'
        if top:              return 'top'
        if bottom:           return 'bottom'
        return None
    def _get_cursor_for_direction(self, direction):
        cursors = {
            'left':         Qt.SizeHorCursor,
            'right':        Qt.SizeHorCursor,
            'top':          Qt.SizeVerCursor,
            'bottom':       Qt.SizeVerCursor,
            'top-left':     Qt.SizeFDiagCursor,
            'bottom-right': Qt.SizeFDiagCursor,
            'top-right':    Qt.SizeBDiagCursor,
            'bottom-left':  Qt.SizeBDiagCursor,
        }
        return cursors.get(direction, Qt.ArrowCursor)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            direction = self._get_resize_direction(event.pos())
            if direction:
                self._resizing = True
                self._resize_direction = direction
                self._resize_start_pos = event.globalPos()
                self._resize_start_geometry = self.geometry()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._resizing = False
        self._resize_direction = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            self._do_resize(event.globalPos())
            event.accept()
            return
        # Update cursor based on hover position
        direction = self._get_resize_direction(event.pos())
        if direction:
            self.setCursor(self._get_cursor_for_direction(direction))
        else:
            self.unsetCursor()

        super().mouseMoveEvent(event)
    def _do_resize(self, global_pos):
        delta = global_pos - self._resize_start_pos
        geo = self._resize_start_geometry
        dx, dy = delta.x(), delta.y()
        min_w, min_h = self.minimumWidth(), self.minimumHeight()

        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        d = self._resize_direction

        if 'right'  in d: w = max(min_w, geo.width()  + dx)
        if 'bottom' in d: h = max(min_h, geo.height() + dy)
        if 'left'   in d:
            w = max(min_w, geo.width()  - dx)
            x = geo.x() + geo.width()  - w
        if 'top'    in d:
            h = max(min_h, geo.height() - dy)
            y = geo.y() + geo.height() - h

        self.setGeometry(x, y, w, h)
    def restore_previous_geometry(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        if self.curr_page == "settings":
            result = self.moveFromSettings(event)
            if result:
                return result
            
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
        result = self.saveSettings()
        if result:
            return

        # self.toggleMenu()
        self.stackedWidget.setCurrentIndex(1)
        self.curr_page = "create"
        
        from widgets.Create import Create 
        page = self.stackedWidget.layout().itemAt(1).widget()
        self.modepage = Create(page)
        self.stackedWidget.setCurrentIndex(1)
        pass
    def moveToAbout(self):
        result = self.saveSettings()
        if result:
            return

        #hiding the nav bar
        # self.toggleMenu()
        self.stackedWidget.setCurrentIndex(3)
        self.curr_page = "about"
        
        pass
    # def moveToHistory(self):
    #     self.curr_page = "history"
    #     pass
    def saveSettings(self):
         if self.curr_page == "settings":
            result = self.moveFromSettings()
            if result:
                return result
            return False
       
    def moveTOSearchArea(self):
        result = self.saveSettings()
        if result:
            return
        self.curr_page = "searchArea"
        # self.toggleMenu()
        self.stackedWidget.setCurrentIndex(5)
        if not hasattr(self, 'search_area_page') or self.search_area_page is None:
            from widgets.SearchArea import SearchArea
            page = self.stackedWidget.layout().itemAt(5).widget()
            self.search_area_page = SearchArea(page)
        self.stackedWidget.setCurrentIndex(5)
        
    def moveToSettings(self):

        if self.curr_page == "settings":
            return

        self.curr_page = "settings"
        # self.toggleMenu()
        self.stackedWidget.setCurrentIndex(6)
        
        if not hasattr(self, 'settings_page') or self.settings_page is None:
            print('before importing settings')
            Settings = get_settings() 
            print('after importing settings')
            page = self.stackedWidget.layout().itemAt(6).widget()
            self.settings_page = Settings(page)
        self.stackedWidget.setCurrentIndex(6)
    def moveFromSettings(self, event = None):
        # print('were changes made', self.settings.value("changesmade"))
        changes = len(self.settings_page.made_changes) > 0 
        print('this is changes', changes)
        if changes:
            reply = QMessageBox.question(self, 'Unsaved Changes', 'You have unsaved changes. Do you want to save them before leaving?', QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if reply == QMessageBox.Save:
                self.settings_page.save_settings()
            elif reply == QMessageBox.Discard:
                self.settings_page.discard_changes()
            elif reply == QMessageBox.Cancel:
                if event is not None:
                    event.ignore()
                return True
        self.settings.sync() 
        return False
      

 
    def moveHome(self):
        # if self.curr_page == "home":
        #    self.toggleMenu()
        result = self.saveSettings()
        if result:
            return

        if self.curr_page == 'create':
            #retrieving the latest data
            # self.toggleMenu()
            # for 
            page = self.stackedWidget.layout().itemAt(0).widget()
            if not hasattr(self, 'homepage') or self.homepage is None:
                print('creating the homepage')
                from widgets.MainPage import MainPage
                self.homepage = MainPage(page)
            else:
                print('updating the homepage')
                self.homepage.refresh_automata_list()
            self.stackedWidget.setCurrentIndex(0)

            
        else:
            self.stackedWidget.setCurrentIndex(0)
            
        self.curr_page = "home"

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
        page = self.stackedWidget.layout().itemAt(0).widget()
        if not hasattr(self, 'homepage') or self.homepage is None:
            from widgets.MainPage import MainPage
            self.homepage = MainPage(page)
        self.stackedWidget.setCurrentIndex(0)
 
    def setupInitialSettings(self):
        print('setting up initial settings')
        import torch
        
        defaults_file = "settings.ini"
        config = configparser.ConfigParser()
        config.read(defaults_file)
        
        for key, value in config.items('general'):
            if self.settings.value(key) is None:
                if value.lower() in ['true', 'false']:
                    self.settings.setValue(key, config.getboolean('general', key))
                elif value.isdigit():
                    self.settings.setValue(key, config.getint('general', key))
                else:
                    try:
                        float_value = float(value)
                        self.settings.setValue(key, float_value)
                    except ValueError:
                        self.settings.setValue(key, value)
        self.settings.setValue('processing', "CPU" if not torch.cuda.is_available() else "GPU")
        self.settings.setValue('cores', max(1, torch.get_num_threads()))
        self.settings.setValue('model', list_downloaded_models()[0]['name'] if len(list_downloaded_models()) > 0 else "")
        
        self.settings.setValue('default_settings', True)
        
        self.settings.sync()
        
        

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
    
    # # Create and show splash screen
    # splash_pix = QPixmap(400, 300)
    # splash_pix.fill(Qt.white)
    
    # # Load and center the logo on the white background
    # logo = QPixmap(resource_path("logo.ico"))
    # if not logo.isNull():
    #     logo = logo.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     from PyQt5.QtGui import QPainter
    #     painter = QPainter(splash_pix)
    #     x = (splash_pix.width() - logo.width()) // 2
    #     y = (splash_pix.height() - logo.height()) // 2 - 30
    #     painter.drawPixmap(x, y, logo)
    #     painter.end()
    
    # splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    # splash.setEnabled(False)
    # splash.showMessage("Loading interface...", Qt.AlignBottom | Qt.AlignHCenter, Qt.black)
    # splash.show()
    # app.processEvents()
    
    # Initialize main window
    window = MainWindow()
    settings.setValue("main_window", window)
    window.show()
    
    # # Close splash and start event loop
    # splash.finish(window)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()



