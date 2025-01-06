from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUi
from util import  Simulate, Message as msg, dataMod, walk
from util.event_tracker import EventTracker
# from util.walk import get_data
import os


# code for mod and after creation of automata
class editAutomata(QDialog):
    def __init__(
        self, data, name
    ):  # data will contain the list of all automata and the name of editing automata
        super(editAutomata, self).__init__()
        ui_path = os.path.join(os.path.dirname(__file__), "../ui/edite.ui")

        # storing the data and name for this specific instance (automata)
        self.data = data
        print('here is the origin data', self.data)
        self.name = name

        loadUi(ui_path, self)

        self.popTable(data, name)
        
        # check if data is being passed

    # def getSelectedRow(self):
    #   curr_row = self.table.currentRow().text() #returns the object of current row
    #   print('curr row',curr_row)
    #   select_row = self.table.currentIndex().row() #returns the index of selected row
    #  # select_row = self.table.removeRow(select_row) #removes the selected row
    #   print(select_row)
    def popTable(self, data, name):

        # editing the title
        self.title.setText("Editing Automata: " + name)

        # data only contains the actions
        # setting the row and name in the table
        self.table.setRowCount(len(data) + 1)

        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("Name"))
        self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(name))
        # populating the rest of the table
        row_index = 1
        # print ("data", data)
        for row in data:
            self.table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row["action"]))
            if row["button"] == "" and row["location"] == [] and row["action"] == "":
                self.table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(""))
            else:
                button_location = [row["button"], '@', row["location"]]
                self.table.setItem(
                    row_index, 1, QtWidgets.QTableWidgetItem(str(button_location))
            )
            row_index += 1

        # adding action event to buttons
        # adding actionlister to button
        self.editButton.clicked.connect(
            lambda checked=False, d=self.data, n=self.name: self.editcell(d, n)
        )

        self.deleteButton.clicked.connect(
            lambda checked=False, d=self.data, n=self.name: self.deleteCell(d, n)
        )
        self.insertButton.clicked.connect(
            lambda checked=False, d=self.data, n=self.name: self.insertCell(d, n)
        )
        # -------------------------------------

        self.simButton.clicked.connect(
            lambda checked=False, d=self.data, n=self.name: self.simulate(d, n)
        )

        self.saveButton.clicked.connect(
            lambda checked=False, d=self.data, n=self.name: self.save(d, n)
        )

        self.runButton.clicked.connect(
            lambda checked=False, d=self.data, n=self.name: self.run(d, n)
        )
        self.cancelButton.clicked.connect(self.cancel)
    def save(self, data=None, name=None):
        # print('here is the data', data)
        name = self.table.item(0, 1).text()
        if name == "Automata":
            msg.warningBox(self, "Error", "Name cannot be 'Automata'")
            return
        #getting the data
        saved_data = walk.get_data()

        # print('saved data', saved_data)


        # print('data', self.data)
        # print('name', self.name)

        if len(saved_data) == 0 :
            # Initialize saved_data if it's None
            print('data is none')
            print('len of data', len(saved_data))
            saved_data = {"Automata": []}

        # Flag to track whether the automata was modified
        mod = False

        # Iterate through existing automata to find a match by name
        for i, automata in enumerate(saved_data.get('Automata', [])):
            if automata['name'] == self.name:
                # Modify the existing automata
                saved_data['Automata'][i] = {"name": self.name, "actions": self.data}
                mod = True
                break

        # If no match was found, add a new automata
        if not mod:
            saved_data['Automata'].append({"name": self.name, "actions": self.data})

        # Write the updated data back to the storage
        walk.write_data(saved_data)

                
        #returning to the welcome page
        # Lazy import
        from widgets.Welcome import Welcome
        
        # Clear current widgets from scroll area
        while self.scrollAreaWidgetContents.layout().count():
            item = self.scrollAreaWidgetContents.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Find the main window
        main_window = self
        while main_window.parent():
            main_window = main_window.parent()
        
        # Assuming main window has a stacked widget or central layout
        if hasattr(main_window, 'centralWidget'):
            # Remove old central widget if it exists
            old_widget = main_window.centralWidget()
            if old_widget:
                old_widget.deleteLater()
                
            # Create and set new Welcome widget
            welcome_widget = Welcome()
            main_window.setCentralWidget(welcome_widget)
        else:
            # Alternative approach if using a layout
            main_layout = main_window.layout()
            if main_layout:
                while main_layout.count():
                    item = main_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                main_layout.addWidget(Welcome())


    def cancel(self):
        
        # Lazy import
        from widgets.Welcome import Welcome
        
        # Clear current widgets from scroll area
        while self.scrollAreaWidgetContents.layout().count():
            item = self.scrollAreaWidgetContents.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Find the main window
        main_window = self
        while main_window.parent():
            main_window = main_window.parent()
        
        # Assuming main window has a stacked widget or central layout
        if hasattr(main_window, 'centralWidget'):
            # Remove old central widget if it exists
            old_widget = main_window.centralWidget()
            if old_widget:
                old_widget.deleteLater()
                
            # Create and set new Welcome widget
            welcome_widget = Welcome()
            main_window.setCentralWidget(welcome_widget)
        else:
            # Alternative approach if using a layout
            main_layout = main_window.layout()
            if main_layout:
                while main_layout.count():
                    item = main_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                main_layout.addWidget(Welcome())
            
    def run(self, data=None, name=None):
        for row in self.data:
            if row['action'] == 'paste':
                Simulate.simPaste("v", True)
            elif row['action'] == 'select all':
                Simulate.simSelectAll(True)
            elif row['action'] == 'click':
                x_coord = row['location'][0]
                y_coord = row['location'][1]
                button = row['button']
                Simulate.simClick(x_coord, y_coord, button, True)
           
        

    def insertCell(self, data=None, name=None):
        row_index = self.table.currentRow()
        if row_index == -1:
            msg.warningBox(self, "Error", "No row selected")
        else:
            # inserting a new row
            self.table.insertRow(row_index + 1)
            #updating the data
            self.data = dataMod.insertRow(self.data, self.name, row_index, {'action':'', 'button': '', 'location': []})
            print('data after insertion', self.data)
            return

    def simulate(self, data=None, name=None):
        cell_index = self.table.currentRow()
        if cell_index == -1:
            msg.warningBox(self, "Error", "Cannot simulate because No row selected")
            return
        else:
            sim = data[cell_index - 1]  # getting the action
            # print(sim)
            if sim["action"] == "click":  # simulating a click
                x_coord = sim["location"][0]
                y_coord = sim["location"][1]
                button = sim["button"]
                # print(f"button: {button} @ location: {x_coord}, {y_coord}")
                Simulate.simClick(x_coord, y_coord, button, True)
            elif sim["action"] == "paste":  # simulating a paste
                Simulate.simPaste("v", True)

    def editcell(self, data=None, name=None):
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
                    newModdata = dataMod.editrow(
                        self.data, name, curr_row - 1, 
                        {"action": type, "button": button, "location": location}
                    )

                    # Update the table visually
                    self.table.setItem(curr_row, 0, QtWidgets.QTableWidgetItem(type))
                    self.table.setItem(curr_row, 1, QtWidgets.QTableWidgetItem(f"{button} @ location: {location}"))

                # Correctly connect the signal with the handler
                event_tracker.tracking_finished.connect(handler_tracking_finished)

                # Start the event tracking thread
                event_tracker.create_new_automaton(True)

            

    def deleteCell(self, data=None, name=None):
        # print('here is the data from delete', data)
        curr_row = self.table.currentRow()
        if curr_row == -1:
            msg.warningBox(self, "Error", "Cannot delete because deleted cell was not selected")
            return
        row_data = self.getCellInfo()
        if row_data[0] == "name":
            msg.warningBox(self, "Error", "Cannot delete the name row")
            return
        else:
            response = msg.questionBox(
                self, "Delete Automata", f"Are you sure you want to delete this row?"
            )
            if response:
                self.table.removeRow(curr_row) #removing selected row
                # updating the data
                self.data = dataMod.deleteRow(
                    self.data, name, curr_row - 1
                )  # curr_row -1 becuse name is  0 index

        
        # print('data after deleting row', self.data)

    def getCellInfo(self):
        # Get the index of the currently selected row
        curr_row = self.table.currentRow()

        if curr_row == -1:
            msg.warningBox(self, "Error", "No row selected")
            return

        # Retrieve all the data from the selected row
        row_data = []
        for column in range(self.table.columnCount()):
            item = self.table.item(curr_row, column)  # Get QTableWidgetItem
            row_data.append(
                item.text() if item else ""
            )  # Get the text or empty string if no item

        return row_data
