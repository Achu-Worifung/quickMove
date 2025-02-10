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
        self.start()

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
        create = loadUi(os.path.join(basedir, './ui/create_automata.ui'))
        self.toggleMenu()
        self.clearWidgets()
        if(self.mainContent.layout().count() > 1):
            print("count is greater than 1")
        else:print("count is less than 1")  
        
        self.mainContent.layout().insertWidget(1,create)
        self.curr_page = "create"
        pass
    def moveToAbout(self):
        #hiding the nav bar
        self.toggleMenu()
        #loading the about page
        about = loadUi(os.path.join(basedir, './ui/about.ui'))
    
        # Deleting the widget at index 1 in the layout
        deleted_pane_item = self.mainContent.layout().takeAt(1)

        # Check if the item at index 1 exists
        self.clearWidgets()
        if(self.mainContent.layout().count() > 1):
            print("count is greater than 1")
        else:print("count is less than 1")    
       

        #adding the about page
        self.mainContent.layout().addWidget(about)
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
            self.clearWidgets()
            if(self.mainContent.layout().count() > 1):
                print("count is greater than 1")
            else:print("count is less than 1")
            self.start(comming_back=True)
            self.mainContent.layout().insertWidget(1,self.home)
        else:
            self.toggleMenu()
            self.clearWidgets()
            self.mainContent.layout().insertWidget(1,self.home)
            
        self.curr_page = "home"
           
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

    def start(self, data=None, comming_back= False):
        if not data:
            data = walk.get_data()

        if not data["Automata"]:
            datapane = self.verticalLayout.layout()
            label = QLabel("No automata found\nClick on the 'Create New Automata' button to create a new automata")
            label.setStyleSheet("font-size: 12pt; color: red;")
            label.setAlignment(Qt.AlignCenter)
            datapane.addWidget(label)
        if comming_back:
            # Get the layout of the home widget
            layout = self.home.layout()

            # If the widget has a layout
            if layout:
                for i in reversed(range(layout.count())):
                    child = layout.itemAt(i).widget()  # Get the child widget
                    if child:
                        child.deleteLater()  # Remove the child widget

            # Adding updated automata to the correct layout of the home widget
            ui_path = os.path.join(os.path.dirname(__file__), './ui/automate_select.ui')

            for i, automaton in enumerate(data['Automata']):
                single_automata = loadUi(ui_path)
                single_automata.name.setText(automaton['name'])

                single_automata.use.clicked.connect(partial(self.getStarted, data, i))
                single_automata.modify.clicked.connect(partial(self.mod, data, automaton['name']))
                single_automata.delete_2.clicked.connect(partial(self.delete, data, automaton['name']))

                # Add the single_automata widget to the home layout
                layout.addWidget(single_automata)

        else:
            datapane = self.created_autos.layout()
            # clearLayout(datapane)

            ui_path = os.path.join(os.path.dirname(__file__), './ui/automate_select.ui')

            for i, automaton in enumerate(data['Automata']):
                single_automata = loadUi(ui_path)
                single_automata.name.setText(automaton['name'])

                # Use partial to capture the current value of i
                single_automata.use.clicked.connect(partial(self.getStarted, data, i))
                single_automata.modify.clicked.connect(partial(self.mod, data, automaton['name']))
                single_automata.delete_2.clicked.connect(partial(self.delete, data, automaton['name']))

                datapane.addWidget(single_automata)

    def delete(self, data=None, button_name=None):
        response = msg.questionBox(self, "Delete Automata", f"Are you sure you want to delete '{button_name}'?")
        
        if response:
            data = walk.get_data()
            data['Automata'] = [auto for auto in data['Automata'] if auto["name"] != button_name]
            
            walk.write_data(data)
            self.start(data, comming_back=True)  # Refresh UI after deletion

    def mod(self, data=None, button_name=None, comming_back= True):
        from widgets.editAutomata import editAutomata

        if not data:
            msg.warningBox(self, "Error", "No data found")
            return

        row_data = next(auto for auto in data["Automata"] if auto["name"] == button_name)
        
        curr_pane = self.scrollPane.layout()
        clearLayout(curr_pane)
        
        edit_pane = editAutomata(row_data['actions'], button_name)
        curr_pane.addWidget(edit_pane)


    def getStarted(self, data=None, index=None):
        search = SearchWidget(data, index)
        
        search_pane = self.scrollPane.layout()
        clearLayout(search_pane)
        search_pane.addWidget(search)

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
