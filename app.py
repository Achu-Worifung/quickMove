import sys
import os
import time
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QRadioButton, QMessageBox, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QRadioButton, QPushButton

import walk  # Placeholder for automata.json data retrieval logic
import event_tracker  # Renamed from Together for clarity


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
            layout = self.widget_2.layout()
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    if widget := item.widget():
                        widget.deleteLater()
            
            layout.addWidget(no_autos)
        else:
            datapane = self.datapane_2.layout()
            # deleteing prev radio buttons
            if datapane is not None:
                while datapane.count():
                    item = datapane.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()  # Delete any existing widgets (radio buttons, buttons, etc.)
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
                
                # Add the radio button and buttons to the pane layout
                pane_layout.addWidget(radio_button)
                pane_layout.addWidget(modify_button)
                pane_layout.addWidget(delete_button)
                
                # Optionally, connect the buttons to their actions
                delete_button.clicked.connect(self.delete)
                modify_button.clicked.connect(self.mod)
                
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

    def mod(self):
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