
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import util.walk as walk
from util.clearLayout import clearLayout
from widgets.SearchWidget import SearchWidget
from functools import partial
from util import event_tracker, Simulate, Message as msg, dataMod
import os
from PyQt5.uic import loadUi
class MainPage(QWidget):
    
    def __init__(self, page_widget):
        super().__init__()
        
        self.page = page_widget
        self.setupUi()
        
    def setupUi(self, data=None):
        if not data:
            data = walk.get_data()

        datapane = self.page.findChild(QWidget, 'created_autos').layout()
        if not data["Automata"]:
            label = QLabel("No automata found\nClick on the 'Create New Automata' button to create a new automata")
            label.setStyleSheet("font-size: 12pt; color: red;")
            label.setAlignment(Qt.AlignCenter)
            datapane.addWidget(label)
        else:
        
            # clearLayout(datapane)

            ui_path = os.path.join(os.path.dirname(__file__), '../ui/automate_select.ui')

            for i, automaton in enumerate(data['Automata']):
                single_automata = loadUi(ui_path)
                single_automata.name.setText(automaton['name'])
                print(automaton)

                # Use partial to capture the current value of i
                single_automata.use.clicked.connect(partial(self.getStarted, data, i))
                single_automata.modify.clicked.connect(partial(self.mod, automaton))
                single_automata.delete_2.clicked.connect(partial(self.delete, data, automaton['name']))

                datapane.addWidget(single_automata)

    def delete(self, data=None, button_name=None):
        response = msg.questionBox(self, "Delete Automata", f"Are you sure you want to delete '{button_name}'?")
        
        if response:
            data = walk.get_data()
            data['Automata'] = [auto for auto in data['Automata'] if auto["name"] != button_name]
            
            walk.write_data(data)
            self.start(data, comming_back=True)  # Refresh UI after deletion

    def mod(self, automaton):
        from widgets.Edit import Edit 
    
        page = self.stackedWidget.layout().itemAt(4).widget()
        self.modepage = Edit(page, automaton)
        self.stackedWidget.setCurrentIndex(4)

    def getStarted(self, data=None, index=None):
        search = SearchWidget(data, index)
        
        search_pane = self.scrollPane.layout()
        clearLayout(search_pane)
        search_pane.addWidget(search)