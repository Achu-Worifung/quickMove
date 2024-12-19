import sys
import os
import time
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QRadioButton, QMessageBox, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QRadioButton, QPushButton
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtWidgets import (
    QApplication, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QDialog, QLabel, QPushButton
)
import dataMod
import Simulate
from PyQt5.QtWidgets import *
import walk  # Placeholder for automata.json data retrieval logic
import event_tracker  # Renamed from Together for clarity

#global method to clear layout
def clearLayout(layout):
    if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()  # Delete any existing widgets (radio buttons, buttons, etc.)
                    elif item.layout():
                        clearLayout(item.layout())  # Recursively clear any layouts
    
class EventTrackerThread(QThread):
    """
    A QThread to handle event tracking without blocking the main UI
    """
    tracking_finished = pyqtSignal(list)
    
    def run(self):
        # Call the tracking function and get results
        action_list = event_tracker.create_new_automaton()
        self.tracking_finished.emit(action_list)

class Welcome(QDialog):
    def __init__(self):

        super(Welcome, self).__init__()

        # Load the UI file
        ui_path = "./ui/intro.ui"
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
            print(datapane)
            #clearing the layout
            clearLayout(datapane)
            
            datapane.addWidget(no_autos)
        else:
            datapane = self.verticalLayout.layout()
            print(datapane)
            
          
            # deleteing prev radio buttons
            clearLayout(datapane)
            
            self.radion_button = []
            for button in data['Automata']:
                # Create a QWidget to hold the radio button and buttons
                pane = QWidget()
                
                # Create a horizontal layout for the radio button and buttons
                pane_layout = QHBoxLayout(pane)
                
                # Create the radio button
                radio_button = QRadioButton(button['name'])
                self.radion_button.append(radio_button)
                
                # Create delete and modify buttons
                delete_button = QPushButton("Delete")
                modify_button = QPushButton("Modify")
                start_button = QPushButton("Start")
                
                # Add the radio button and buttons to the pane layout
                pane_layout.addWidget(radio_button)
                pane_layout.addWidget(modify_button)
                pane_layout.addWidget(delete_button)
                pane_layout.addWidget(start_button)
                
                # Optionally, connect the buttons to their actions
                delete_button.clicked.connect(self.delete)
                modify_button.clicked.connect(lambda checked=False, d=data, b=button['name']: self.mod(d, b))
 #so much easier to do this( passsing data on a click event)
                
             
                
                # Add the pane to the datapane layout
                datapane.addWidget(pane)

    def delete(self):
        button = self.sender() #getting the button triggering the event
        parent = button.parent() #getting the parent of the button
        automata_name = parent.findChildren(QRadioButton)[0].text() #getting the text of the radio button
        
        print(f"Delete button clicked for {parent.findChildren(QRadioButton)[0].text()}")
        # print(f"Delete button clicked for {button.parent()}")
        response = QMessageBox.question(self, "Delete Automata", f"Are you sure you want to delete '{automata_name}'?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if response == QMessageBox.Yes:
            data = walk.get_data()
            # Filter out the automata with the given name
            data["Automata"] = [automaton for automaton in data["Automata"] if automaton["name"] != automata_name]
            print(f"Data: {data}")
            
            #overwrite json with new data
            walk.write_data(data)
            
            self.start(data) # calling the start method to update the UI (inefficient but it works)
    def mod(self, data = None, button_name = None):
        if not data:
            QMessageBox.warning(self, "Error", "No data passed to modify automata")
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
            edit.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f'{[actions['button'], actions["location"]]}'))
        
        edit.title.setText("Editing Automata: " + button_name)
        datapane.addWidget(edit)
        
        
        #adding actionlister to button 
        edit.editButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.editcell(d, n))
        
        edit.deleteButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.deleteCell(d, n))
        
        edit.insertButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.insertCell(d, n))
        #-------------------------------------
        
        edit.simButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.simulate(d, n))
        
        edit.saveButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.deleteCell(d, n))
        
        edit.runButton.clicked.connect(lambda checked=False, d=data, n=button_name: edit.deleteCell(d, n))
                    

        
        
        # Function to load data into the table
    
        
        
      

                

# code for mod and after creation of automata
class editAutomata(QDialog):
    def __init__(self, data, name): #data will contain the list of all automata and the name of editing automata
        super(editAutomata, self).__init__()
        ui_path = "./ui/edite.ui"
        loadUi(ui_path, self)
        #check if data is being passed
    # def getSelectedRow(self):
    #   curr_row = self.table.currentRow().text() #returns the object of current row
    #   print('curr row',curr_row)
    #   select_row = self.table.currentIndex().row() #returns the index of selected row
    #  # select_row = self.table.removeRow(select_row) #removes the selected row
    #   print(select_row)
      
    def run(self):
        #running the entire automata
        for row in range(self.table.rowCount()):
            pass
    def insertCell(self, data = None, name = None):
        row_index = self.table.currentRow()
        if row_index == -1:
            QMessageBox.warning(self, "Error", "No row selected")
            return
        else:
            #inserting a new row
            self.table.insertRow(row_index+1)
            return
    def simulate(self, data = None, name = None):
        cell_index = self.table.currentRow()
        if cell_index == -1:
            QMessageBox.warning(self, "Error", "No row selected")
            return
        else:
            sim = data['Automata'][2]['actions'][cell_index-1] #getting the action
            print(sim)
            if sim['action'] == 'click': #simulating a click
                x_coord = sim['location'][0]
                y_coord = sim['location'][1]
                button = sim['button']
                print(f'button: {button} @ location: {x_coord}, {y_coord}')
                Simulate.simClick(x_coord, y_coord, button, True)
            elif sim['action'] == 'paste': #simulating a paste
                Simulate.simPaste('v', True)
    def editcell(self, data = None, name = None):
        curr_row = self.table.currentRow()
        #editting the name of the automata
        if curr_row == -1:
            QMessageBox.warning(self, "Error", "No row selected")
            return
        elif(curr_row == 0):
            text, ok = QInputDialog.getText(self, "Rename Automata", "Enter new name:")
            if ok:
                self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(text))
        else:
            response = QMessageBox.question(self, "Edit Cell", f"Click Yes to edit cell and record 1 I/O event.",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if response == QMessageBox.Yes:
                event_tracker.create_new_automaton(True)
                #call the event tracker to record only 1 event
                event = event_tracker.create_new_automaton(True)
                print('here is the event',event) #nothing is being returned
                type = event[0]['action']
                button = event[0]['button']
                location = event[0]['location']
                dataMod.editrow(data, name, curr_row-1, {"action": type, "button": button, "location": location})
                #updating the table
                self.table.setItem(curr_row,0, QtWidgets.QTableWidgetItem(type))
                self.table.setItem(curr_row,1, QtWidgets.QTableWidgetItem(f'{button} @ location:{location}'))
                
    def deleteCell(self, data= None, name = None):
        curr_row = self.table.currentRow()
        row_data = self.getCellInfo()
        if(row_data[0] == 'name'):
            QMessageBox.warning(self, "Error", "Cannot delete the name row")
            return
        if curr_row == -1:
            QMessageBox.warning(self, "Error", "No row selected")
            return
        else:
            response = QMessageBox.question(self, "Delete Automata", f"Are you sure you want to delete this row?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if response == QMessageBox.Yes:
                self.table.removeRow(curr_row)
                #updating the data 
                dataMod.deleteRow(data, name, curr_row-1) #curr_row -1 becuse name is  0 index         
            else:
                return
    def getCellInfo(self):
         # Get the index of the currently selected row
        curr_row = self.table.currentRow()
        
        if curr_row == -1:
            QMessageBox.warning(self, "Error", "No row selected")
            return

        # Retrieve all the data from the selected row
        row_data = []
        for column in range(self.table.columnCount()):
            item = self.table.item(curr_row, column)  # Get QTableWidgetItem
            row_data.append(item.text() if item else "")  # Get the text or empty string if no item

        return row_data
    def insertBelow(self):
        pass
       
            
class noAutomata(QDialog):
    def __init__(self):
        super(noAutomata, self).__init__()
        ui_path = "./ui/no_autos.ui"
        loadUi(ui_path, self)
        #adding acuion event to button
        self.create_auto.clicked.connect(self.start_tracking)
        
    def start_tracking(self):
        # Show tracking started message
        start = QMessageBox.information(
            self,
            'Start Tracking',
            'Press Ok to start tracking and ESC to stop',
            QMessageBox.StandardButton.Ok
        )   
        if start == QMessageBox.StandardButton.Ok:
            # start tracking
            print("Tracking started")
            

    # def loaddata(self):
    #     # Clear any existing widgets
    #     for i in reversed(range(self.scroll_layout.count())): 
    #         self.scroll_layout.itemAt(i).widget().setParent(None)

    #     # Mock implementation of walk.get_data
    #     data = walk.get_data()
        
    #     print(f"Data: {data}, Count: {len(data)}")

    #     if not data:
    #         # If no automata, display a message
    #         no_data_label = QtWidgets.QLabel("No automata available.")
    #         self.scroll_layout.addWidget(no_data_label)
            
    #         # Create button
    #         button = QtWidgets.QPushButton("Create", self)
    #         self.scroll_layout.addWidget(button)
            
    #         # Add action listener to the button
    #         button.clicked.connect(self.start_tracking)
    #     else:
    #         # Add a radio button for each automaton
    #         for auto in data:
    #             radio_button = QRadioButton(auto.get("name"))
    #             self.scroll_layout.addWidget(radio_button)

    #             # Connect the radio button to a slot
    #             radio_button.toggled.connect(
    #                 lambda checked, name=auto.get("Automata_1"): self.radio_selected(checked, name)
    #             )

    # def radio_selected(self, checked, name):
    #     if checked:  # Respond only when the button is selected
    #         print(f"Selected automaton: {name}")
    
    def start_tracking(self):
        # Show tracking started message
        msg = QMessageBox()
        msg.setWindowTitle("Tracking")
        msg.setText("Tracking started \nYour mouse and keyboard events are being tracked\nPress 'Esc' to stop tracking")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

        # Create and start the event tracker thread
        self.event_tracker_thread = EventTrackerThread()
        self.event_tracker_thread.tracking_finished.connect(self.on_tracking_finished)
        self.event_tracker_thread.start()

    def on_tracking_finished(self, action_list):
        print(f"Action List: {action_list}")
        
        json_data = {"Automata": [
            {
                "name": "Automata",
                "actions": action_list
            },
            {
                "name": "Automata1",
                "actions": action_list
            },
            {
                "name": "Automata2",
                "actions": action_list
            }
        ]}
        
        print(f"JSON Data: {json_data}")
        # Optionally save or process the action list
        
        # You might want to call a method in walk.py to save the actions
        walk.write_data(json_data)
        if action_list:
            QMessageBox.information(self, "Tracking Complete", 
                                    f"Tracked {len(action_list)} actions.")
        

# Main application
def main():
    app = QApplication(sys.argv)
    welcome = Welcome()
    welcome.show()
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Exiting due to: {e}")

if __name__ == "__main__":
    main()