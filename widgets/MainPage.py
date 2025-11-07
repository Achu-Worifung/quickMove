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
        self.data = []
        
    def setupUi(self, data=None):
        # Ensure we don't crash if the page/widget is missing
        created_widget = self.page.findChild(QtWidgets.QWidget, 'created_autos')
        if created_widget:
            # Clear any existing items safely
            if created_widget.layout() is not None:
                clearLayout(created_widget.layout())
        else:
            # Nothing to populate
            return

        # Prefer explicitly passed data, otherwise load persisted data
        if data is not None:
            self.data = data
        else:
            self.data = walk.get_data() or {}

        # Ensure we have a layout to add to; create one if necessary
        datapane = created_widget.layout()
        if datapane is None:
            datapane = QVBoxLayout()
            created_widget.setLayout(datapane)

        automata_list = self.data.get('Automata', [])
        if not automata_list:
            return

        ui_path = os.path.join(os.path.dirname(__file__), '../ui/automate_select.ui')

        for i, automaton in enumerate(automata_list):
            try:
                single_automata = loadUi(ui_path)
            except Exception:
                # if the .ui file is missing or loadUi fails, skip this entry
                continue

            # Safely set fields and connect buttons if they exist
            if hasattr(single_automata, 'name'):
                single_automata.name.setText(automaton.get('name', ''))

            if hasattr(single_automata, 'use'):
                single_automata.use.clicked.connect(partial(self.getStarted, automaton=automaton))
            if hasattr(single_automata, 'modify'):
                single_automata.modify.clicked.connect(partial(self.mod, automaton))
            if hasattr(single_automata, 'delete_2'):
                single_automata.delete_2.clicked.connect(partial(self.delete, button_name=automaton.get('name')))

            datapane.addWidget(single_automata)
        

    def delete(self, button_name=None):
        
        if not self.data:
            self.data = walk.get_data()
       
        response = msg.questionBox(self, "Delete Automata", f"Are you sure you want to delete '{button_name}'?")
        
        if response:
            index_to_remove = None
            for i, auto in enumerate(self.data['Automata']):
                if auto["name"] == button_name:
                    index_to_remove = i
                    break
            self.data['Automata'] = [auto for auto in  self.data['Automata'] if auto["name"] != button_name]
            
            
            
            walk.write_data( self.data)
            

            layout = self.page.findChild(QWidget, 'created_autos').layout()
            item = layout.itemAt(index_to_remove)
            if item:
                widget = item.widget()
                if widget:
                    layout.removeWidget(widget)
                    widget.deleteLater()
    def mod(self, automaton):
        from widgets.Edit import Edit 
        
        # print(automaton)
    
        edit_page = self.page.parent().layout().itemAt(4).widget()
        self.modpage = Edit(edit_page, automaton)
        self.page.parent().setCurrentIndex(4)

    def getStarted(self, automaton):
        from widgets.SearchWidget import SearchWidget
        search_page = self.page.parent().layout().itemAt(2).widget()
        search_page = SearchWidget(search_page=search_page, data = automaton, index = 2)
        self.page.parent().setCurrentIndex(2)
