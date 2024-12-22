from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUi
from util import event_tracker, Simulate, Message as msg, dataMod
from util.clearLayout import clearLayout
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

        print("len of data ", len(data))
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("Name"))
        self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(name))
        # populating the rest of the table
        row_index = 1
        for row in data:
            self.table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row["action"]))
            button_location = [row["button"], '@', row["location"]]
            self.table.setItem(
                row_index, 1, QtWidgets.QTableWidgetItem(str(button_location))
            )
            row_index += 1

        # adding action event to buttons
        # adding actionlister to button
        self.editButton.clicked.connect(
            lambda checked=False, d=data, n=name: self.editcell(d, n)
        )

        self.deleteButton.clicked.connect(
            lambda checked=False, d=data, n=name: self.deleteCell(d, n)
        )

        self.insertButton.clicked.connect(
            lambda checked=False, d=data, n=name: self.insertCell(d, n)
        )
        # -------------------------------------

        self.simButton.clicked.connect(
            lambda checked=False, d=data, n=name: self.simulate(d, n)
        )

        self.saveButton.clicked.connect(
            lambda checked=False, d=data, n=name: self.deleteCell(d, n)
        )

        self.runButton.clicked.connect(
            lambda checked=False, d=data, n=name: self.run(d, n)
        )
        self.cancelButton.clicked.connect(self.cancel)
    def cancel(self):
        # Lazy import
        from widgets.Welcome import Welcome
        
        # Clear current widgets from scroll area
        while self.scrollAreaWidgetContents.layout().count():
            item = self.scrollAreaWidgetContents.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Find the main window (assuming it's the top-level parent)
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
        # running the entire automata
        #    sim = data['Automata'][2]['actions'] #getting the action list why 2
        for action in data:
            if action["action"] == "click":  # simulating a click
                x_coord = action["location"][0]
                y_coord = action["location"][1]
                button = action["button"]
                print(f"button: {button} @ location: {x_coord}, {y_coord}")
                Simulate.simClick(x_coord, y_coord, button, True)
            elif action["action"] == "paste":  # simulating a paste
                Simulate.simPaste(action["text"])

    def insertCell(self, data=None, name=None):
        row_index = self.table.currentRow()
        if row_index == -1:
            msg.warningBox(self, "Error", "No row selected")
        else:
            # inserting a new row
            self.table.insertRow(row_index + 1)
            return

    def simulate(self, data=None, name=None):
        cell_index = self.table.currentRow()
        if cell_index == -1:
            msg.warningBox(self, "Error", "No row selected")
            return
        else:
            sim = data[cell_index - 1]  # getting the action
            # print(sim)
            if sim["action"] == "click":  # simulating a click
                x_coord = sim["location"][0]
                y_coord = sim["location"][1]
                button = sim["button"]
                print(f"button: {button} @ location: {x_coord}, {y_coord}")
                Simulate.simClick(x_coord, y_coord, button, True)
            elif sim["action"] == "paste":  # simulating a paste
                Simulate.simPaste("v", True)

    def editcell(self, data=None, name=None):
        curr_row = self.table.currentRow()
        # editting the name of the automata
        if curr_row == -1:
            msg.warningBox(self, "Error", "No row selected")
            return
        elif curr_row == 0:
            text, ok = msg.inputDialogBox(self, "Rename Automata", "Enter new Name:")
            if ok:
                self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(text))
        else:
            response = msg.questionBox(
                self, "Edit Cell", f"Click Yes to edit cell and record 1 I/O event."
            )

            if response:
                event_tracker.create_new_automaton(True)
                # call the event tracker to record only 1 event
                event = event_tracker.create_new_automaton(True)
                print("here is the event", event)  # nothing is being returned
                type = event[0]["action"]
                button = event[0]["button"]
                location = event[0]["location"]
                dataMod.editrow(
                    data,
                    name,
                    curr_row - 1,
                    {"action": type, "button": button, "location": location},
                )
                # updating the table
                self.table.setItem(curr_row, 0, QtWidgets.QTableWidgetItem(type))
                self.table.setItem(
                    curr_row,
                    1,
                    QtWidgets.QTableWidgetItem(f"{button} @ location:{location}"),
                )

    def deleteCell(self, data=None, name=None):
        print('here is the data from delete', data)
        curr_row = self.table.currentRow()
        row_data = self.getCellInfo()
        if row_data[0] == "name":
            msg.warningBox(self, "Error", "Cannot delete the name row")
            return
        if curr_row == -1:
            msg.warningBox(self, "Error", "No row selected")
            return
        else:
            response = msg.questionBox(
                self, "Delete Automata", f"Are you sure you want to delete this row?"
            )
            if response:
                self.table.removeRow(curr_row) #removing selected row
                # updating the data
                data = dataMod.deleteRow(
                    data, name, curr_row - 1
                )  # curr_row -1 becuse name is  0 index
            else:
                return

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
