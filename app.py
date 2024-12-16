import sys
import os
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QRadioButton, QMessageBox, QWidget
from PyQt5.QtCore import QThread, pyqtSignal

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

    def start(self):
        data = walk.get_data()

        if not data:
            no_autos = noAutomata()
            
            # Clear current layout and add no_autos widget
            layout = self.widget_3.layout()
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    if widget := item.widget():
                        widget.deleteLater()
            
            layout.addWidget(no_autos)

            
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