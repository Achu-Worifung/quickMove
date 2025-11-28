from PyQt5.QtWidgets import QLabel, QPushButton, QWidget, QTableWidget
from PyQt5 import QtWidgets
from util import  Simulate, Message as msg, dataMod, walk
from util.event_tracker import EventTracker
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QSettings
class Edit(QWidget):
    def __init__(self, page_widget, data= None):
        super().__init__()
        self.page_widget = page_widget
        self.data = data
        # Now we can safely update content
        self.updated_data = data['actions']
        self.name = data['name']
        updateRequested = pyqtSignal(dict, bool)
        
        
        self.settings = QSettings("MyApp", "AutomataSimulator")

        
        # Set up widget references
        self.setup_widgets()
        
        
        # Now we can safely update content
        # self.updatecontend("Hello")
    
    def setup_widgets(self):
        """Get references to all widgets on the page"""
       
        self.title = self.page_widget.findChild(QLabel, 'title_2')  # or whatever widget type it is
        self.run_button = self.page_widget.findChild(QPushButton, 'runButton')
        self.edit_button = self.page_widget.findChild(QPushButton, 'editButton')
        self.insert_button = self.page_widget.findChild(QPushButton, 'insertButton')
        self.save_button= self.page_widget.findChild(QPushButton, 'saveButton')
        self.sim_button= self.page_widget.findChild(QPushButton, 'simButton')
        self.del_button= self.page_widget.findChild(QPushButton, 'deleteButton')
        self.can_button= self.page_widget.findChild(QPushButton, 'cancelButton')
        self.table= self.page_widget.findChild(QTableWidget, 'table')
        self.barLocation = self.page_widget.findChild(QPushButton, 'barButton')
        
        print('bar location', self.barLocation)
        
        #adding action to the buttons
       
        #-----------------------------------------------------------------------  
        self.run_button.clicked.connect(self.run)
        self.edit_button.clicked.connect(self.edit)
        self.insert_button.clicked.connect(self.insert)
        self.save_button.clicked.connect(self.save)
        self.sim_button.clicked.connect(self.simulate)
        self.can_button.clicked.connect(self.cancel)
        self.del_button.clicked.connect(self.delete)
        self.barLocation.clicked.connect(self.setBarLocation)
        #-----------------------------------------------------------------------
        #populating the table
        self.title.setText(self.data['name']) #setting the title
        
        self.table.setRowCount(len(self.data['actions']) + 1)
        
        #placing the title at the top of the table
        
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("Name"))
        self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(self.data['name']))
        # print('here is the data', self.data)
        row_index = 1
        for row in self.data['actions']:
            self.table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row["action"]))
            if row["button"] == "" and row["location"] == [] and row["action"] == "":
                self.table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(""))
            else:
                button_location = [row["button"], '@', row["location"]]
                self.table.setItem(
                    row_index, 1, QtWidgets.QTableWidgetItem(str(button_location))
            )
            row_index += 1
        
        
        
    def run(self):
        for row in self.updated_data:
            if row['action'] == 'paste':
                Simulate.simPaste("v", True)
            elif row['action'] == 'select all':
                Simulate.simSelectAll(True)
            elif row['action'] == 'click':
                x_coord = row['location'][0]
                y_coord = row['location'][1]
                button = row['button']
                Simulate.simClick(x_coord, y_coord, button, True)
    def edit(self):
        curr_row = self.table.currentRow()
        
        if curr_row == -1:
            msg.warningBox(self, "Error", "No row selected")
            return
        elif curr_row == 0:
            text, ok = msg.inputDialogBox(self, "Rename Automata", "Enter new Name:")
            if ok:
                self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(text))
                self.name = text
                self.title.setText("Editing Automata: " + text)
        else:
            response = msg.questionBox(self, "Edit Cell", "Click Yes to edit cell and record 1 I/O event.")
            
            if response:
                event_tracker = EventTracker()
                
                # Define the handler function inline and properly connect it
                def handler_tracking_finished(event):
                    # print('Event received:', event)
                    event_data = event[0]
                    type = event_data["action"]
                    button = event_data["button"]
                    location = event_data["location"]
                    
                    # Ensure data modification is applied
                    self.updated_data = dataMod.editrow(
                        self.updated_data, self.name, curr_row - 1, 
                        {"action": type, "button": button, "location": location}
                    )

                    # Update the table visually
                    self.table.setItem(curr_row, 0, QtWidgets.QTableWidgetItem(type))
                    self.table.setItem(curr_row, 1, QtWidgets.QTableWidgetItem(f"{button} @ location: {location}"))

                # Correctly connect the signal with the handler
                event_tracker.tracking_finished.connect(handler_tracking_finished)

                # Start the event tracking thread
                event_tracker.create_new_automaton(True)  
    def insert(self):
        row_index = self.table.currentRow()
        if row_index == -1:
            msg.warningBox(self, "Error", "No row selected")
        else:
            # inserting a new row
            self.table.insertRow(row_index + 1)
            #updating the data
            self.updated_data = dataMod.insertRow(self.updated_data, self.name, row_index, {'action':'', 'button': '', 'location': []})
            # print('data after insertion', self.data)
            return
    def save(self):
        # print("here is teh data", self.updated_data)
        # print("here is the name", self.name)
        # return
        name = self.table.item(0, 1).text()
        if name == "Automata":
            msg.warningBox(self, "Error", "Name cannot be 'Automata'")
            return
        #getting the data
        saved_data = walk.get_data()


        if len(saved_data) == 0 :
            # Initialize saved_data if it's None
            # print('data is none')
            # print('len of data', len(saved_data))
            saved_data = {"Automata": []}

        # Flag to track whether the automata was modified
        mod = False

        # Iterate through existing automata to find a match by name
        for i, automata in enumerate(saved_data.get('Automata', [])):
            if automata['name'] == self.name:
                # Modify the existing automata
                saved_data['Automata'][i] = {"name": self.name, "actions": self.updated_data}
                mod = True
                break

        # If no match was found, add a new automata
        if not mod:
            saved_data['Automata'].append({"name": self.name, "actions": self.updated_data})

        # Write the updated data back to the storage
        walk.write_data(saved_data)
        
        # --------------RETURNING TO THE MAIN PAGE----------------
        from widgets.MainPage import MainPage
    
        main = self.page_widget.parent().layout().itemAt(0).widget()
        self.modepage = MainPage(main)
        # self.stackedWidget.setCurrentIndex(0)
       
        self.page_widget.parent().setCurrentIndex(0)
    def simulate(self):
        cell_index = self.table.currentRow()
        if cell_index == -1:
            msg.warningBox(self, "Error", "Cannot simulate because No row selected")
            return
        else:
            sim = self.updated_data[cell_index - 1]  # getting the action
            # print(sim)
            if sim["action"] == "click":  # simulating a click
                x_coord = sim["location"][0]
                y_coord = sim["location"][1]
                button = sim["button"]
                # print(f"button: {button} @ location: {x_coord}, {y_coord}")
                Simulate.simClick(x_coord, y_coord, button, True)
            elif sim["action"] == "paste":  # simulating a paste
                Simulate.simPaste("v", True)
    def cancel(self):
        #-------------------------------------------------------
        #using the parent to get the stack widget
        print(self.page_widget.parent())
        self.page_widget.parent().setCurrentIndex(0)
    def delete(self):
        curr_row = self.table.currentRow()
        if curr_row == -1:
            msg.warningBox(self, "Error", "Cannot delete because deleted cell was not selected")
            return
        row_data = curr_row
        print(row_data)
        if row_data== 0:
            msg.warningBox(self, "Error", "Cannot delete the name row")
            return
        else:
            response = msg.questionBox(
                self, "Delete Automata", f"Are you sure you want to delete this row?"
            )
            if response:
                self.table.removeRow(curr_row) #removing selected row
                # updating the data
                self.updated_data = dataMod.deleteRow(
                    self.updated_data, self.name, curr_row - 1
                )  # curr_row -1 becuse name is  0 index

    def setBarLocation(self):  
       
        cell_index = self.table.currentRow()
        if cell_index == -1:
            msg.warningBox(self, "Error", "Cannot simulate because No row selected")
            return
        else:
            sim = self.updated_data[cell_index - 1]  # getting the action
            # print(sim)
            self.settings.setValue(self.title.text(), sim["location"])
            print(self.settings.value(self.title.text()))        
