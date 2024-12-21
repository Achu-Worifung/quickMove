from PyQt5 import QtWidgets
import os
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem,QWidget
from PyQt5.uic import loadUi
from util import event_tracker,Simulate, Message as msg, dataMod
from PyQt5.QtWidgets import (
    QApplication, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QDialog, QLabel, QPushButton,QHBoxLayout, QRadioButton, QPushButton
)
# from widgets import noAutomata, editAutomata
from widgets.noAutomata import noAutomata
from widgets.editAutomata import editAutomata
from util import clearLayout, walk
# from util.walk import get_data
from util.clearLayout import clearLayout
class Welcome(QDialog):
    def __init__(self):

        super(Welcome, self).__init__()

        # Load the UI file
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/intro.ui')

        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"
        loadUi(ui_path, self)

        # # Initialize scroll area layout
        # if not self.scrollAreaWidgetContents.layout():
        #     self.scroll_layout = QVBoxLayout(self.scrollAreaWidgetContents)
        # else:
        #     self.scroll_layout = self.scrollAreaWidgetContents.layout()

        # Initialize event tracker thread
        self.event_tracker_thread = None

        # Load data into the UI
        # self.loaddata()
         # Call the start method
        self.start()

    def start(self, data = None):
      
           
        if not data:
            data = walk.get_data()
        
        if not data:
            no_autos = noAutomata()
            
            # Clear current layout and add no_autos widget
            datapane = self.scrollPane.layout()
            # print(datapane)
            #clearing the layout
            clearLayout(datapane)
            
            datapane.addWidget(no_autos)
        else:
            datapane = self.verticalLayout.layout()
            print(datapane)
            
          
            # deleteing prev radio buttons
            clearLayout(datapane)
            
            self.radion_button = []
            #pane for each automata
            link = os.path.join(os.path.dirname(__file__), '../ui/automate_select.ui')

            for button in data['Automata']:
                single_automata = loadUi(link)#loading the ui file
                single_automata.name.setText(button['name'])
                single_automata.modify.clicked.connect(lambda checked=False, d=data, b=button['name']: self.mod(d, b))

                single_automata.delete_2.clicked.connect(lambda checked=False, d=data, b=button['name']: self.delete(d, b))

#                 # Create a QWidget to hold the radio button and buttons
#                 pane = QWidget()
                
#                 # Create a horizontal layout for the radio button and buttons
#                 pane_layout = QHBoxLayout(pane)
                
#                 # Create the radio button
#                 radio_button = QRadioButton(button['name'])
#                 self.radion_button.append(radio_button)
                
#                 # Create delete and modify buttons
#                 delete_button = QPushButton("Delete")
#                 modify_button = QPushButton("Modify")
#                 start_button = QPushButton("Start")
                
#                 # Add the radio button and buttons to the pane layout
#                 pane_layout.addWidget(radio_button)
#                 pane_layout.addWidget(modify_button)
#                 pane_layout.addWidget(delete_button)
#                 pane_layout.addWidget(start_button)
                
#                 # Optionally, connect the buttons to their actions
#                 delete_button.clicked.connect(self.delete)
#                 modify_button.clicked.connect(lambda checked=False, d=data, b=button['name']: self.mod(d, b))
#  #so much easier to do this( passsing data on a click event)
                
             
                
                # Add the pane to the datapane layout
                datapane.addWidget(single_automata)

    def delete(self, data = None, button_name = None):
        # button = self.sender() #getting the button triggering the event
        # parent = button.parent() #getting the parent of the button
        # automata_name = parent.findChildren(QRadioButton)[0].text() #getting the text of the radio button
        
        # print(f"Delete button clicked for {parent.findChildren(QRadioButton)[0].text()}")
        # print(f"Delete button clicked for {button.parent()}")
        response = msg.questionBox(self, "Delete Automata", f"Are you sure you want to delete '{button_name}'?")
        
        if response:
            data = walk.get_data()
            # Filter out the automata with the given name
            # data["Automata"] = [automaton for automaton in data["Automata"] if automaton["name"] != button_name]
            # print(f"Data: {data}")
            
            index = 0
            #deleting the automata
            for i in range(len(data['Automata'])):
                if data['Automata'][i]['name'] == button_name:
                    data['Automata'].pop(i)
                    index = i
                    break
            
            #overwrite json with new data
            walk.write_data(data)

            #getting the automata
            auto = self.verticalLayout.layout().takeAt(index)
            if auto:
                widget = auto.widget()
                if widget:  # Ensure the widget exists before removing and deleting it
                    widget.setParent(None)
                    widget.deleteLater()
                else:
                    print(f"No widget found at index {index}")
            else:
                print(f"No item at index {index}")

            
            # self.start(data) # calling the start method to update the UI (inefficient but it works)
            
            #issue several button that edit the data but each button access to the data doesn't affect the other
    def mod(self, data = None, button_name = None):
        if not data:
            msg.warningBox(self, "Error", "No data found")
            return 
        
        edit = editAutomata(data, button_name)
        
        # Clear current layout and add no_autos widget
        datapane = self.scrollPane.layout()
        clearLayout(datapane)

        # Find the specific automaton by button_name
        row_data = [automaton for automaton in data["Automata"] if automaton["name"] == button_name][0]
        
        # Set the number of rows to match the number of actions plus the additional row
        edit.table.setRowCount(len(row_data['actions']) + 1)
        
        # Add the additional row first
        edit.table.setItem(0, 0, QtWidgets.QTableWidgetItem('name'))
        edit.table.setItem(0, 1, QtWidgets.QTableWidgetItem(button_name))
        
        # Populate the table with actions, starting from the next row
        for row, actions in enumerate(row_data['actions'], start=1):
            edit.table.setItem(row, 0, QtWidgets.QTableWidgetItem(actions['action']))
            button_location = [actions['button'], actions['location']]
            edit.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(button_location)))

        
        edit.title.setText("Editing Automata: " + button_name)
        datapane.addWidget(edit)
        
        
        #adding actionlister to button 
        edit.editButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.editcell(d, n))
        
        edit.deleteButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.deleteCell(d, n))
        
        edit.insertButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.insertCell(d, n))
        #-------------------------------------
        
        edit.simButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.simulate(d, n))
        
        edit.saveButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.deleteCell(d, n))
        
        edit.runButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.run(d, n))
                    

        
        
        # Function to load data into the table
    
        
        
      