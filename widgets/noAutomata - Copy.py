from PyQt5 import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from util import event_tracker, walk, Message as msg
import util.clearLayout as clearLayout
from widgets.editAutomata import editAutomata
import os
class noAutomata(QDialog):
    def __init__(self, label = None):
        super(noAutomata, self).__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/no_autos.ui')

        loadUi(ui_path, self)
        #adding action event to button
        self.create_auto.clicked.connect(self.start_tracking)
        if label:
            self.label_4.setText("Create new automata")
        
    # def start_tracking(self):
    #     # Show tracking started message
    #     start = msg.informationBox(self, "StartTracking", 'Press Ok to start tracking and ESC to stop')
        
    #     if start :
    #         # start tracking
    #         print("Tracking started")
    
    def start_tracking(self):
        # Show tracking started message
        ok = msg.questionBox(self, "StartTracking", 'Press YES to start tracking and STOP to stop')
        
        if ok :
            # start tracking
            print("Tracking started")
            # Create and start the event tracker thread
            self.event_tracker_thread = EventTrackerThread()
            self.event_tracker_thread.tracking_finished.connect(self.on_tracking_finished)
            self.event_tracker_thread.start()


    def on_tracking_finished(self, action_list):
        mainwindow = self.parent().layout()
        mainwindow.addWidget(editAutomata(action_list, 'Automata'))



class EventTrackerThread(QThread):
    """
    A QThread to handle event tracking without blocking the main UI
    """
    tracking_finished = pyqtSignal(list)
    
    def run(self):
        # Call the tracking function and get results
        action_list = event_tracker.create_new_automaton()
        self.tracking_finished.emit(action_list)