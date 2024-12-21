from PyQt5 import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from util import event_tracker, walk, Message as msg
import os
class noAutomata(QDialog):
    def __init__(self):
        super(noAutomata, self).__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/no_autos.ui')

        loadUi(ui_path, self)
        #adding acuion event to button
        self.create_auto.clicked.connect(self.start_tracking)
        
    def start_tracking(self):
        # Show tracking started message
        start = msg.informationBox(self, "StartTracking", 'Press Ok to start tracking and ESC to stop')
        
        if start :
            # start tracking
            print("Tracking started")
    
    def start_tracking(self):
        # Show tracking started message
        start = msg.informationBox(self, "StartTracking", 'Press Ok to start tracking and ESC to stop')
        
        if start :
            # start tracking
            print("Tracking started")
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
        #check this out later
        # if action_list:
        #     QMessageBox.information(self, "Tracking Complete", 
        #                             f"Tracked {len(action_list)} actions.")


class EventTrackerThread(QThread):
    """
    A QThread to handle event tracking without blocking the main UI
    """
    tracking_finished = pyqtSignal(list)
    
    def run(self):
        # Call the tracking function and get results
        action_list = event_tracker.create_new_automaton()
        self.tracking_finished.emit(action_list)