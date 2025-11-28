from PyQt5.QtWidgets import QPushButton, QWidget, QPlainTextEdit
from PyQt5.QtCore import QThread
from util.event_tracker import EventTracker
class Create(QWidget):
    def __init__(self, page_widget):
        super().__init__()
        self.page_widget = page_widget
       
       
        self.setupWidget()
        self.tracking = False
        
    
    def setupWidget(self):
        #--------------INITIALIZING THE WIDGET'S COMPOONENT----------------------
        self.record = self.page_widget.findChild(QPushButton, 'record')
        self.stop = self.page_widget.findChild(QPushButton, 'stop')
        self.text_edit= self.page_widget.findChild(QPlainTextEdit, 'plainTextEdit')
        
        #--------------ADDING ACTION TO THE BUTTONS-------------------------------
        self.record.clicked.connect(self.start_tracking)
        self.stop.clicked.connect(self.stop_tracking)
        
        
        
        
        
    def start_tracking(self):
        if self.tracking:
            return
        from util import   Message as msg
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
       
        # modifying the action list to remove last click
        self.event_tracker_thread.tracker.actionList = self.event_tracker_thread.tracker.actionList[:-1]
        self.event_tracker_thread.tracker.stop_tracking()
        
    def on_event_recorded(self, event):
        """Handle new events as they come in"""
        self.text_edit.appendPlainText("New event recorded: {event}.\n".format(event=event))

    def on_tracking_finished(self, action_list):
        #-----------------------------------------------------------------------
        # print(action_list)
        
        #--------------------------GOING TO EDIT PAGE TO EDIT NEWLY CREATED AUTOMATA----------------------
        
        #-----------------GETTING THE STACK-------------------
        stack = self.page_widget.parent()
        # -----------------SETTING THE STACK TO THE EDIT PAGE----------------
        stack.setCurrentIndex(4)
        
        # ---------------MOVING TO EDIT PAGE-------------------
        from widgets.Edit import Edit
        page = stack.currentWidget()
        data = {'name': 'New Automata', 'actions': action_list}
        self.modepage = Edit(page, data)
        
        #clearing the ui
        # page = self.stackedWidget.layout().itemAt(4).widget()
        # self.modepage = Edit(page, automaton)
        # self.page_widget.parent().setCurrentIndex(4)
        # mainwindow.addWidget(Edit(action_list, 'Automata'))
        
class EventTrackerThread(QThread):
    def __init__(self):
        super().__init__()
        self.tracker = EventTracker()
        
    def run(self):
        self.tracker.create_new_automaton()