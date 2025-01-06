# noauto.py
from PyQt5 import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt5.uic import loadUi
from util import walk, Message as msg

from widgets.editAutomata import editAutomata
import os
from util.event_tracker import EventTracker  # Update import path as needed

class noAutomata(QDialog):
    def __init__(self, label=None):
        super(noAutomata, self).__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/create_automata.ui')
        loadUi(ui_path, self)
        
        # Initialize UI
        self.record.clicked.connect(self.start_tracking)
        self.tracking = False
        #making sure stop cannot be clicked before tracking is started
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.stop_tracking)

    def start_tracking(self):
        if self.tracking:
            return
        ok = msg.questionBox(self, "StartTracking", 'Press YES to start tracking and ESC to stop\n tip: for hotkeys like crtl+a press a followed by crtl')
        
        if ok:
            print("Tracking started")
            self.event_tracker_thread = EventTrackerThread()
            self.event_tracker_thread.tracker.event_recorded.connect(self.on_event_recorded)
            self.event_tracker_thread.tracker.tracking_finished.connect(self.on_tracking_finished)
            self.event_tracker_thread.start()
            self.tracking = True
            self.stop.setEnabled(True)
    def stop_tracking(self):
       
        #modifying the action list to remove last click
        self.event_tracker_thread.tracker.actionList = self.event_tracker_thread.tracker.actionList[:-1]
        self.event_tracker_thread.tracker.stop_tracking()
        
    def on_event_recorded(self, event):
        """Handle new events as they come in"""
        self.plainTextEdit.appendPlainText("New event recorded: {event}.\n".format(event=event))

    def on_tracking_finished(self, action_list):
        from util.clearLayout import clearLayout
        """Handle completion of tracking"""
        print("Tracking finished")
        mainwindow = self.layout()
        clearLayout(mainwindow)
        mainwindow.addWidget(editAutomata(action_list, 'Automata'))

class EventTrackerThread(QThread):
    def __init__(self):
        super().__init__()
        self.tracker = EventTracker()
        
    def run(self):
        self.tracker.create_new_automaton()