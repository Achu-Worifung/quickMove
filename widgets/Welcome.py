from PyQt5 import QtWidgets
import os
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem,QWidget
from PyQt5.uic import loadUi
from util import event_tracker,Simulate, Message as msg, dataMod
from PyQt5.QtWidgets import (
    QApplication, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QDialog, QLabel, QPushButton,QHBoxLayout, QRadioButton, QPushButton, QMainWindow
)
from PyQt5.QtCore import QSettings

# from widgets import noAutomata, editAutomata
from widgets.SearchWidget import SearchWidget
from PyQt5.QtCore import Qt
# from util import clearLayout, walk
import util.walk as walk
# from util.walk import get_data
from util.clearLayout import clearLayout
class Welcome(QMainWindow):
    def __init__(self):

        super(Welcome, self).__init__()

        # Load the UI file
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/MainWindow.ui')

        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"
        loadUi(ui_path, self)
        #removing default window controls
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.settings = QSettings("MyApp", "AutomataSimulator")

        # Initialize event tracker thread
        self.event_tracker_thread = None
        # self.search_area.clicked.connect(self.configure_search_area)

        # adding functions to all the buttons
        self.all_buttons_function();
         # Call the start method
        self.start()

    def start(self, data = None):   
        if not data:
            autos = self.settings.value("Automatas", [])
        
        if not data:
            datapane = self.verticalLayout.layout()
            label = QLabel("No automata found\nClick on the 'Create New Automata' button to create a new automata")
            label.setStyleSheet("font-size: 12pt; color: red;") # Changed em to px
            label.setAlignment(Qt.AlignCenter)  # Changed self.label to label
            datapane.addWidget(label)
            
 
        else:
            datapane = self.verticalLayout.layout()
            
          
            # deleteing prev radio buttons
            clearLayout(datapane)
            
            #pane for each automata
            link = os.path.join(os.path.dirname(__file__), '../ui/automate_select.ui')

            for i, button in data['Automata']:
                single_automata = loadUi(link)#loading the ui file
                
                single_automata.use.clicked.connect(lambda checked = False, d = data, index = i: self.getStarted(d,  i))
                
                single_automata.name.setText(button['name'])
                
                single_automata.modify.clicked.connect(lambda checked=False, d=data, b=button['name']: self.mod(d, b))
            

                single_automata.delete_2.clicked.connect(lambda checked=False, d=data, b=button['name']: self.delete(d, b))
                
             
                
                # Add the pane to the datapane layout
                datapane.addWidget(single_automata)
        
    def delete(self, data = None, button_name = None):
        # button = self.sender() #getting the button triggering the event
        # parent = button.parent() #getting the parent of the button
        # automata_name = parent.findChildren(QRadioButton)[0].text() #getting the text of the radio button
        
        # print(f"Delete button clicked for {parent.findChildren(QRadioButton)[0].text()}")
        # print(f"Delete button clicked for {button.parent()}")
        response = msg.questionBox(self, "Delete Automata", f"Are you sure you want to delete '{button_name}'?")
        
        if response:
            data = walk.get_data()
            # Filter out the automata with the given name
            # data["Automata"] = [automaton for automaton in data["Automata"] if automaton["name"] != button_name]
            # print(f"Data: {data}")
            
            index = 0
            #deleting the automata
            for i in range(len(data['Automata'])):
                if data['Automata'][i]['name'] == button_name:
                    data['Automata'].pop(i)
                    index = i
                    break
            
            #overwrite json with new data
            walk.write_data(data)

            #getting the automata
            auto = self.verticalLayout.layout().takeAt(index)
            if auto:
                widget = auto.widget()
                if widget:  # Ensure the widget exists before removing and deleting it
                    widget.setParent(None)
                    widget.deleteLater()
                

            
            # self.start(data) # calling the start method to update the UI (inefficient but it works)
            
            #issue several button that edit the data but each button access to the data doesn't affect the other
    def mod(self, data = None, button_name = None): #data = entire json file
        from widgets.editAutomata import editAutomata
        
        if not data:
            msg.warningBox(self, "Error", "No data found")
            return 

        # Find the specific automaton by button_name
        row_data = [automaton for automaton in data["Automata"] if automaton["name"] == button_name][0]
        
        
        # Clear current layout and add no_autos widget
        curr_pane = self.scrollPane.layout()
        clearLayout(curr_pane)

        #providing edit automata with the data and the name of the automata
        edit_pane = editAutomata(row_data['actions'], button_name)
        curr_pane.addWidget(edit_pane)


    def all_buttons_function(self):
        #closing button
        closing = self.
                    

    def createAutomata(self):
        from widgets.noAutomata import noAutomata

        no_autos = noAutomata('Create new automata')
            
        # Clear current layout and add no_autos widget
        datapane = self.scrollPane.layout()
        # print(datapane)
        #clearing the layout
        clearLayout(datapane)
            
        datapane.addWidget(no_autos)
        
        # Function to load data into the table
    
    def getStarted(self, data = None, index = None):
        
        #clearing the welcome page and moving to the search
        search = SearchWidget(data, index)
            
        # Clear current layout and add no_autos widget
        search_pane = self.scrollPane.layout()
        #clearing the layout
        clearLayout(search_pane)
        
        search_pane.addWidget(search)
    
    

    def configure_search_area(self):
        #use event tracker to establish the search area
        from util.event_tracker import EventTracker
        msg.informationBox(self, 'search area', "Configure the searcharea. Click Ok to get started.1st click on the top left corner of the search area\n2nd click on the bottom right corner of the search area\nPress 'ESC' to stop")
        def setup_search_area(event):
            print('here is the event ', event)
            if len(event) >=2:
                left = event[0]['location'][0]
                top = event[0]['location'][1]
                right = event[1]['location'][0]
                bottom = event[1]['location'][1]
                print(left, top, right, bottom)
            else:
                msg.warningBox(self, 'Error', 'Please select the search area')
                
                return
            # Save the search area to QSettings
            self.settings = QSettings("MyApp", "AutomataSimulator")
            self.settings.setValue("search_area", tuple([left, top, right, bottom]))
            msg.informationBox(self, 'search area', "Search area configured successfully")



            # pass
        self.event_tracker_thread = EventTracker()
        self.event_tracker_thread.tracking_finished.connect(setup_search_area)
        self.event_tracker_thread.create_new_automaton()
        
      