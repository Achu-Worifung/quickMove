import os
import sys
from dotenv import load_dotenv
from PyQt5.uic import loadUi
import httpx
from PyQt5.QtWidgets import QDialog, QApplication, QLabel
from collections import OrderedDict
import util.getReference as getReference
from widgets.SearchBar import AutocompleteWidget
from PyQt5.QtCore import QThread, pyqtSignal, QMimeData, QTimer
import asyncio
from functools import partial
from typing import Dict, Optional
import util.Simulate as Simulate
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
import util.savedVerses as savedVerses
from PyQt5.QtCore import QSettings
from urllib.parse import quote_plus
import re
import string
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *


class Tracker():
    def __init__(self, reference, result, priority):
        super().__init__()
        self.reference = reference
        self.result = result
        self.priority = priority
    def __repr__(self):
        # This representation is used when printing lists of objects
        return f"Tracker(reference='{self.reference}', priority={self.priority})"

class SearchThread(QThread):
    finished = pyqtSignal(list, str)

    def __init__(self, api_key: str, engine_id: str, query: str):
        super().__init__()
        self.api_key = api_key
        self.engine_id = engine_id
        self.query = query

    def run(self):
        # print("quote_plus(self.query)", quote_plus(self.query))
        url = 'https://customsearch.googleapis.com/customsearch/v1?'
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": quote_plus(self.query  ),  
            "safe": "active",  # Optional
            'hl': 'en',
            'num': 10,
            
            
        }
        response = httpx.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        # print('results for real', results)
        reversed_items = results.get("items", [])[::-1]
        # print('results reversed hello', reversed_items)
        self.finished.emit(reversed_items, self.query)
        # print('here are the items',results.get("items", []))

class locateVerseThread(QThread):
    finished = pyqtSignal(str)

    def run(self):
        import util.findVerseBox as findVerseBox
        self.finished.emit(findVerseBox.findPrevDisplayedVerse())


        

class SearchWidget(QDialog):
    def __init__(self, search_page, data=None, index=None):
        super().__init__()
        
        self.search_page = search_page
        
        self.init_page()
       
        # self.name = name
        self.old_widget= []
        self.saved_widgets = []
        self.version.addItems(['','KJV', 'NIV', 'ESV'])
        #list of saved verse
        self.savedVerse = []
        self.Tracker = []
        self.pop = True

        self.version.setCurrentIndex(0)
        self.data = data
        self.name = data['name']
        # print('data', data)
        self.verse_tracker = OrderedDict()
        self.verse_widgets: Dict[str, QDialog] = {}  # Store widget references
        self.search_thread: Optional[SearchThread] = None
        self.locate_box_thread: Optional[locateVerseThread] = None
        self.last_query_time = 0
        self.displayed_verse = []

        #initialize qsetting to store clipboard history
        self.settings = QSettings("MyApp", "AutomataSimulator")
        self.basedir = self.settings.value("basedir")
        
        
        load_dotenv(self.basedir + '/.env')
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('SEARCH_ENGINE_ID')

        self.pop_saved_verse()
        self.pop = False
        self.double_search = True #if no result found, search again
        
        self.init_page()
        from widgets.SearchBar import AutocompleteWidget

        self.autoComplete_widget = AutocompleteWidget(self.search_bar[0])
        self.search_bar[0].textChanged.connect(
            lambda text, d=self.data: self.handle_search(d, text)
        )
        # customLineEdit = autoComplete_widget.lineedit
        # adding an infocuse listerner to get the location of the search bar
        self.autoComplete_widget.lineedit.focusOutEvent = self.get_prev_verse  
        self.autoComplete_widget.lineedit.focusInEvent = self.openEye
        self.prevVerse.clicked.connect(lambda checked=False:   QTimer.singleShot(0, lambda: Simulate.present_prev_verse(self.name, self.data)))
        
        # autoComplete_widget.lineedit.focusInEvent = self.openEye

    def openEye(self, event):
       
        QLineEdit.focusInEvent(self.autoComplete_widget.lineedit, event)
        self.searchBarContainer.findChild(QPushButton).setIcon(QIcon(self.basedir + '/Icons/open.ico'))
    def init_page(self):
        self.prevVerse = self.search_page.findChild(QPushButton, 'prev_verse_3')
        self.version = self.search_page.findChild(QComboBox, 'version_3')
        self.searchPane = self.search_page.findChild(QVBoxLayout, 'searchPane_3')
        self.savedVerse_pane = self.search_page.findChild(QVBoxLayout, 'saved_verse_pane')
        self.search_bar = self.search_page.findChildren(QLineEdit)
        self.history = self.search_page.findChild(QLabel, 'history_4')
        
        self.searchBarContainer = self.search_page.findChild(QWidget, 'widget_15')
        # self.searchBarContainer.mousePressEvent = self.open
        # self.searchBarContainer.click
        
        print('search bar', self.search_bar)

    #this function will run after the search bar is in focus
    def get_prev_verse(self, event):
        # return # Disable for now
        # Ensure the thread is only started once

        
        if self.locate_box_thread is None or not self.locate_box_thread.isRunning():
            self.locate_box_thread = locateVerseThread()
            self.locate_box_thread.finished.connect(self.handle_locate_results)
            self.locate_box_thread.start()
            super().focusOutEvent(event)
         # Call the original focusOutEvent to maintain the default behavior
        QLineEdit.focusOutEvent(self.autoComplete_widget.lineedit, event)
        self.searchBarContainer.findChild(QPushButton).setIcon(QIcon(self.basedir + '/Icons/close.ico'))
    def handle_locate_results(self, result):
        """
        Slot to process the results from locateVerseThread.
        """
        

        # Remove '@' from the result and strip extra spaces
        result = re.sub(r'@', '', result).strip()
        print('result 1', result)

        # Remove all non-printable characters from the result
        result = ''.join(char for char in result if char in string.printable)

        print('result 2', result)
        # removing the Bible reference using regex
        new_result = re.sub(r'\).*$', '', result).strip()
        #saving the retrieved result 
        self.settings.setValue('displayed_reference', new_result)

        # Display the result with history information
        self.updateHistoryLabel()


    def updateHistoryLabel(self):
        #getting the values
        original = self.settings.value('copied_reference')
        displayed = self.settings.value('displayed_reference')
        self.history.setText(f'Original Reference: {original} Currently Displayed Verse: <b>{displayed}</b>')
        #clearing the values to avoid persistence
        # self.settings.setValue('copied_reference', '').
        # self.settings.setValue('displayed_reference', '')
        
        
    
    #so when a verse result is clicked the current verse references is saved to clipboard
    #we have location of x and y of highlighted verse box
    def prevVerse(self):
       #presenting the previous verse
       print('data', self.data)
       Simulate.present_prev_verse(self.name)
       #updaing the label to show the current verse
       self.updateHistoryLabel()



    def pop_saved_verse(self):
        
        get_saved_verse = savedVerses.getSavedVerses()
        if not get_saved_verse:
            self.savedVerse = {'savedVerses': []}
            return

        # print('get_saved_verse', get_saved_verse)
        verses = get_saved_verse['savedVerses']
        self.savedVerse = get_saved_verse
        # print('verses', verses)
        if verses:
            for verse in verses:
                self.savedVerses(verse['title'], verse['body'])

    
    def delete_saved_verse(self, title):
        data = self.savedVerse['savedVerses']
        new_list = [verse for verse in data if verse['title'] != title]
        self.savedVerse['savedVerses'] = new_list
        savedVerses.saveVerse(self.savedVerse)

    def add_saved_verse(self, title, body):
        # print('adding saved verse', self.savedVerse)
        new_saved_verse = self.savedVerse['savedVerses']
        new_saved_verse.insert(0,{'title': title, 'body': body})
        self.savedVerse['savedVerses'] = new_saved_verse
        savedVerses.saveVerse(self.savedVerse)


    def update_verse_tracker(self, new_results, query):
        # print('updating verse tracker')


        widget_to_remove = []
        #clearing the layout of old results
        for i in range(self.searchPane.count()):
            widget_to_remove.append(self.searchPane.itemAt(i).widget())
        for widget in widget_to_remove:
            self.searchPane.removeWidget(widget)
            widget.deleteLater()
        self.old_widget.clear()
        self.displayed_verse.clear()
        print('current contedents', self.searchPane.count())
        #adding the new results to the layout
        self.searchPane.update()
        for result in new_results:
            reference = getReference.getReference(result['title'])
            # print('diplayed verses', self.displayed_verse)
            if not reference:
                continue
            #ensuring reference are uniqu
            if reference in self.displayed_verse:
                #placing it at the top
                index = self.displayed_verse.index(reference)
                remove_widget = self.searchPane.takeAt(index)
                if remove_widget.widget():
                    remove_widget.widget().deleteLater()
                    remove_widget.widget().setParent(None)
                    # self.searchPane.removeItem(remove_widget)


            else:
                self.displayed_verse.append(reference)
            self.add_verse_widget(query, result, reference)
        
    

            

       
#add verse widget to the searchPane
    def add_verse_widget(self,query, result, reference):
      
        #sorting based on priority
        print('result', result)
       
        link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
        single_result = loadUi(link)
        
        body = getReference.boldedText(result['snippet'], query)
        single_result.body.setText(body)
        single_result.title.setText(reference)
        # print('verse_key', verse_key)
        
        single_result.save.clicked.connect(lambda checked=False,t=reference, b=body: self.savedVerses(t, b))
        self.searchPane.insertWidget(0, single_result)
        self.old_widget.append(single_result)
        # self.verse_widgets[verse_key] = single_result
        def mouse_click(event, verse_key):
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))

        
        # Bind `verse_key` explicitly
        single_result.title.mousePressEvent = partial(mouse_click, verse_key=reference)
        single_result.body.mousePressEvent = partial(mouse_click, verse_key=reference)
        
        # self.searchPane.single_result.clicked.connect(self.present)  

    def handle_search(self, data=None, query=None):
        # print('data', data)
        if query == "":
            self.double_search = True
            print('erasing query')
            
            # Clear the displayed verses and searchPane widgets
            self.displayed_verse.clear()
            print('current contents', self.searchPane.count())

            widgets_to_remove = []
            for i in range(self.searchPane.count()):
                widget = self.searchPane.itemAt(i).widget()
                if widget:
                    widgets_to_remove.append(widget)
            
            # Remove and delete widgets
            for widget in widgets_to_remove:
                self.searchPane.removeWidget(widget)
                widget.deleteLater()
            
            self.old_widget.clear()
            self.verse_tracker = OrderedDict()
            self.old_widget = []
            self.saved_widgets = []
            self.search_thread = None
            self.last_query_time = 0
            print('after erasing', self.searchPane.count())
            self.searchPane.update()
            return


        # Check if the query is valid
        if not query or query.count(' ') < 3 or query[-1] != ' ':
            self.double_search = True
            return

        # Cancel any existing search
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.terminate()
            self.search_thread.wait()

        # Start new search thread
        self.search_thread = SearchThread(self.api_key, self.engine_id, query)
        self.search_thread.finished.connect(
            lambda results, q=query: self.handle_search_results(results, q)
        )
        self.search_thread.start()

    def handle_search_results(self, results, query):
        # if we have results update the verse tracker
        if results:
            self.update_verse_tracker(results, query)
        elif not results and self.searchPane.count() > 0 and self.double_search: #if we have no results and nothing is displayed
            print('no results')
            # self.double_search = False
            # print('double search ')
            # self.handle_search(self.data, query = query + 'KJV')

    #function to add saved verses to the saved pane
    def savedVerses(self, title, body):
        # current_size = len(self.verse_widgets)
        # saved_verses_len = len(self.saved_verse)
        old_result_len = len (self.old_widget)
        link = os.path.join(os.path.dirname(__file__), '../ui/saved.ui')
        saved = loadUi(link)
        saved.title.setText(title)
        saved.body.setText(body)
        saved.delete_2.clicked.connect(lambda checked=False: self.delete(saved))
        self.saved_widgets.append(saved)
        #add one to len latter
        
        #adding action listener to title and body
        def mouse_click(event, verse_key):
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))

           
        # Bind `verse_key` explicitly
        saved.title.mousePressEvent = partial(mouse_click, verse_key=title)
        saved.body.mousePressEvent = partial(mouse_click, verse_key=title)

        self.savedVerse_pane.insertWidget(0, saved)
        if self.pop:
            return
        self.add_saved_verse(title, body)
        
    #deleting function for saved verses
    def delete(self, saved):
        # print('deleting', saved.title.text())
        #finding the saved pane and deleting it
        index = self.savedVerse_pane.indexOf(saved)
        if index != -1:  # Widget found
            item = self.savedVerse_pane.takeAt(index)
            if item.widget():
                item.widget().deleteLater()
                self.delete_saved_verse(saved.title.text())
        pass
    def present(self, title, automata):
        #getting the originator of the event
        # clicked_widget = self.sender()
        # verse_reference = clicked_widget.title.text()
        # clipboard = QApplication.clipboard()
        # clipboard.setText(title)
        #storing the verse to the qsetting clipboard history
        self.settings.setValue("next_verse", title)
        #setting up the clipboard
        prev_action = ''
        
        # print('here is the action list', automata)
        
        # for action in automata['actions']:
        #     if(action['action'] == 'click'):
        #         print('clicking')
        #     elif(action['action'] == 'paste'):
        #         print('pasting')
        #     elif(action['action'] == 'select all'):
        #         print('selecting all')
        # return
        #using simulating the automata
        for action in automata['actions']:
            # print('prev action', prev_action)

            if action['action'] == 'click':
                x_coord = action['location'][0]
                y_coord = action['location'][1]
                button = action['button']
                Simulate.simClick(x_coord, y_coord, button)
                

            elif action['action'] == 'paste':
                print('pasting')
                Simulate.simPaste(title, True)

            elif action['action'] == 'select all':
                Simulate.simSelectAll(True)

        #updating the history label
        self.updateHistoryLabel()
    #change auto function
   