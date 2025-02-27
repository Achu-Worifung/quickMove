
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
        #deleteing anything that is in the layout
        widget = self.page.findChild(QWidget, 'created_autos')
        if widget:
            print("in widget setup",widget.layout())
            clearLayout(widget.layout())
        # print('set up ran')
        if not data:
            self.data = walk.get_data()

        datapane = self.page.findChild(QWidget, 'created_autos').layout()
        if not  self.data["Automata"]:
           pass
        else:
        
            # clearLayout(datapane)

            ui_path = os.path.join(os.path.dirname(__file__), '../ui/automate_select.ui')

            for i, automaton in enumerate( self.data['Automata']):
                single_automata = loadUi(ui_path)
                single_automata.name.setText(automaton['name'])
                # print(automaton)

                # Use partial to capture the current value of i
                single_automata.use.clicked.connect(partial(self.getStarted, automaton = automaton))
                single_automata.modify.clicked.connect(partial(self.mod, automaton))
                single_automata.delete_2.clicked.connect(partial(self.delete, button_name = automaton['name']))

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
      