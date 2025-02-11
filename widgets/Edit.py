from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from functools import partial
class Edit(QWidget):
    def __init__(self, page_widget, data= None):
        super().__init__()
        self.page_widget = page_widget
        self.data = data
        print(self.data)
        
        # Set up widget references
        self.setup_widgets()
        
        
        # Now we can safely update content
        self.updatecontend("Hello")
    
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
        
        #adding action to the buttons
       
        #-----------------------------------------------------------------------  
        self.run_button.clicked.connect(self.run)
        self.edit_button.clicked.connect(self.edit)
        self.insert_button.clicked.connect(self.insert)
        self.save_button.clicked.connect(self.save)
        self.sim_button.clicked.connect(self.simulate)
        self.can_button.clicked.connect(self.cancel)
        self.del_button.clicked.connect(self.delete)
        #-----------------------------------------------------------------------
        #populating the table
        self.title.setText(self.data['name']) #setting the title
        
        self.table.setRowCount(len(self.data['actions']) + 1)
        
        #placing the title at the top of the table
        
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("Name"))
        self.table.setItem(0, 1, QtWidgets.QTableWidgetItem(self.data['name']))
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
        
        
        self.check()
        
    def run(self):
        print("run")
    def edit(self):
        print("edit")    
    def insert(self):
        print("insert")
    def save(self):
        print("save")
    def simulate(self):
        print("simulate")
    def cancel(self):
        print("cancel")
    def delete(self):
        print("delete")
    def check(self):
        print(self.run_button.text())
        print(self.edit_button.text())
        print(self.insert_button.text())
        print(self.save_button.text())
        print(self.sim_button.text())
        print(self.can_button.text())
        print(self.table)
        
    def updatecontend(self, hllo):
        if hasattr(self, 'title'):  # Safe check
            self.title.setText(hllo)
        else:
            print("Warning: title widget not found")