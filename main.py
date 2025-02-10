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

    def start(self, data=None):   
        if not data:
            data = walk.get_data()

        if not data["Automata"]:
            datapane = self.verticalLayout.layout()
            label = QLabel("No automata found\nClick on the 'Create New Automata' button to create a new automata")
            label.setStyleSheet("font-size: 12pt; color: red;")
            label.setAlignment(Qt.AlignCenter)
            datapane.addWidget(label)
        else:
            datapane = self.created_autos.layout()
            clearLayout(datapane)

            ui_path = os.path.join(os.path.dirname(__file__), './ui/automate_select.ui')

            for i, automaton in enumerate(data['Automata']):
                single_automata = loadUi(ui_path)
                single_automata.name.setText(automaton['name'])
                
                single_automata.use.clicked.connect(lambda checked=False, d=data, index=i: self.getStarted(d, index))
                single_automata.modify.clicked.connect(lambda checked=False, d=data, b=automaton['name']: self.mod(d, b))
                single_automata.delete_2.clicked.connect(lambda checked=False, d=data, b=automaton['name']: self.delete(d, b))
                
                datapane.addWidget(single_automata)

    def delete(self, data=None, button_name=None):
        response = msg.questionBox(self, "Delete Automata", f"Are you sure you want to delete '{button_name}'?")
        
        if response:
            data = walk.get_data()
            data['Automata'] = [auto for auto in data['Automata'] if auto["name"] != button_name]
            
            walk.write_data(data)
            self.start(data)  # Refresh UI after deletion

    def mod(self, data=None, button_name=None):
        from widgets.editAutomata import editAutomata

        if not data:
            msg.warningBox(self, "Error", "No data found")
            return

        row_data = next(auto for auto in data["Automata"] if auto["name"] == button_name)
        
        curr_pane = self.scrollPane.layout()
        clearLayout(curr_pane)
        
        edit_pane = editAutomata(row_data['actions'], button_name)
        curr_pane.addWidget(edit_pane)

    def createAutomata(self):
        from widgets.noAutomata import noAutomata

        no_autos = noAutomata('Create new automata')

        datapane = self.scrollPane.layout()
        clearLayout(datapane)
        datapane.addWidget(no_autos)

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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
