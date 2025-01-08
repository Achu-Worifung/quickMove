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




class SearchThread(QThread):
    finished = pyqtSignal(dict, str)

    def __init__(self, api_key: str, engine_id: str, query: str):
        super().__init__()
        self.api_key = api_key
        self.engine_id = engine_id
        self.query = query

    def run(self):
        url = 'https://customsearch.googleapis.com/customsearch/v1'
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": quote_plus(self.query),  
            "siteSearch": "biblegateway.com",  # Optional
            # "safe": "active",  # Optional
        }

        # Request with proper headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        response = httpx.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        self.finished.emit(results, self.query)
        print('here are the items',results.get("items", []))

class locateVerseThread(QThread):
    finished = pyqtSignal(str)

    def run(self):
        import util.findVerseBox as findVerseBox
        self.finished.emit(findVerseBox.findPrevDisplayedVerse())


class SearchWidget(QDialog):
    def __init__(self, data=None):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/search.ui')
        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"
        loadUi(ui_path, self)

        self.version.addItems(['','KJV', 'NIV', 'ESV'])
        #list of saved verse
        self.savedVerse = []
        self.pop = True

        self.version.setCurrentIndex(0)
        self.data = data
        # print('data', data)
        self.verse_tracker = OrderedDict()
        self.verse_widgets: Dict[str, QDialog] = {}  # Store widget references
        self.search_thread: Optional[SearchThread] = None
        self.locate_box_thread: Optional[locateVerseThread] = None
        self.last_query_time = 0
        
        autoComplete_widget = AutocompleteWidget(self)
        autoComplete_widget.setStyleSheet('height: 50px; border-radius: 10px; font-size: 20px;')
        autoComplete_widget.lineedit.setPlaceholderText('Search for a verse')
        #adding an unfocused listerner to get the location of the search bar
        autoComplete_widget.lineedit.focusOutEvent = self.get_prev_verse  #disableing this for now
        self.horizontalLayout.addWidget(autoComplete_widget)
        autoComplete_widget.lineedit.textChanged.connect(
            lambda text, d=self.data: self.handle_search(d, text)
        )
         #adding action listerner to change auto button
        self.pushButton.clicked.connect(lambda checked=False: self.changeAuto())

        self.prev_verse.clicked.connect(lambda checked=False:   QTimer.singleShot(0, lambda: Simulate.present_prev_verse()))

        #initialize qsetting to store clipboard history
        self.settings = QSettings("MyApp", "AutomataSimulator")

        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('SEARCH_ENGINE_ID')

        self.pop_saved_verse()
        self.pop = False

    #this function will run after the search bar is unfocused
    def get_prev_verse(self, event):
        # return # Disable for now
        # Ensure the thread is only started once
        if self.locate_box_thread is None or not self.locate_box_thread.isRunning():
            self.locate_box_thread = locateVerseThread()
            self.locate_box_thread.finished.connect(self.handle_locate_results)
            self.locate_box_thread.start()

        # Call the default focusOutEvent
        # autoComplete_widget.lineedit.focusOutEvent(self.autoComplete_lineedit, event)

    def handle_locate_results(self, result):
        """
        Slot to process the results from locateVerseThread.
        """
        

          # Displaying the previous and next verse
        next_verse = self.settings.value('next_verse')
        prev_verse = self.settings.value('prev_verse')
        print(f'Previous verse: {prev_verse} Next verse: {next_verse}')

        # Remove '@' from the result and strip extra spaces
        result = re.sub(r'@', '', result).strip()

        # Remove all non-printable characters from the result
        result = ''.join(char for char in result if char in string.printable)

        # removing the Bible reference using regex
        new_result = re.sub(r'\s*\([A-Z]{2,5}\)\s*', '', result).strip()
        
        #ensure that new_result and next verse are the same book and chapt

        # Get the number after the colon
        match = re.search(r':(\d+)', new_result)
        if match:
            verse_number = match.group(1)  # Keeps the verse number as a string
            print('Verse number:', verse_number)
           
            self.settings.setValue('verse_num', verse_number)
          
        else:
            self.settings.setValue('verse_num', None)
            print('No verse number found in new_result')

        # print('New result:', new_result)

        # Display the result with history information
        self.history.setText(f'Original Reference: {next_verse} Currently Displayed Verse: {new_result}')



    
    #so when a verse result is clicked the current verse references is saved to clipboard
    #we have location of x and y of highlighted verse box
    def prevVerse(self):

       Simulate.present_prev_verse()



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
        print('adding saved verse', self.savedVerse)
        new_saved_verse = self.savedVerse['savedVerses']
        new_saved_verse.insert(0,{'title': title, 'body': body})
        self.savedVerse['savedVerses'] = new_saved_verse
        savedVerses.saveVerse(self.savedVerse)


    def update_verse_tracker(self, new_results, query):
        # print('verse tracker', self.verse_tracker, '\n\n\n')
        # print('verse widgets', self.verse_widgets, '\n\n\n')
        # print('top ten results', new_results[:10])

        new_results = new_results[::-1]
        new_keys = []

        for result in new_results:
            title_array = getReference.getReference(result['title'])
            if not title_array:
                continue
            verse_key = title_array[0]
            new_keys.append(verse_key)

            if verse_key in self.verse_widgets:
                # Update existing widget content
                body = getReference.boldedText(result['snippet'], query)
                self.verse_widgets[verse_key].body.setText(body)
            else:
                # Add new widget for the verse
                self.add_verse_widget(verse_key, result, query)

        # Remove widgets not in the latest results
        for verse_key in list(self.verse_widgets.keys()):
            if verse_key not in new_keys:
                widget = self.verse_widgets.pop(verse_key)
                self.searchPane.removeWidget(widget)
                widget.deleteLater()

    def add_verse_widget(self, verse_key, result, query):
        link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
        single_result = loadUi(link)
        
        body = getReference.boldedText(result['snippet'], query)
        single_result.body.setText(body)
        single_result.title.setText(verse_key)
        # print('verse_key', verse_key)
        
        single_result.save.clicked.connect(lambda checked=False,t=verse_key, b=body: self.savedVerses(t, b))
        self.searchPane.insertWidget(0, single_result)
        self.verse_widgets[verse_key] = single_result
        def mouse_click(event, verse_key):
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))

           
        # Bind `verse_key` explicitly
        single_result.title.mousePressEvent = partial(mouse_click, verse_key=verse_key)
        single_result.body.mousePressEvent = partial(mouse_click, verse_key=verse_key)
        
        # self.searchPane.single_result.clicked.connect(self.present)  

    def handle_search(self, data=None, query=None):
        if query == "":
            # Keep track if we've found the label
            found_label = False
            # Store widgets to delete
            widgets_to_delete = []
            
            # First pass: identify widgets to delete
            for i in range(self.searchPane.count()):
                widget = self.searchPane.itemAt(i).widget()
                if isinstance(widget, QLabel):
                    found_label = True
                    continue
                if not found_label and widget:
                    widgets_to_delete.append(widget)
            
            # Second pass: delete the identified widgets
            for widget in widgets_to_delete:
                print('deleting widget', widget)
                self.searchPane.removeWidget(widget)
                widget.deleteLater()
            self.verse_tracker = OrderedDict()
            self.verse_widgets: Dict[str, QDialog] = {}  # Store widget references
            self.search_thread: Optional[SearchThread] = None
            self.last_query_time = 0
            return
        # print('here is the query', query)

        if not query or query.count(' ') < 3 or query[len(query)-1] != ' ':
            return

        # Cancel any existing search
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.terminate()
            self.search_thread.wait()

        # Start new search thread
        self.search_thread = SearchThread(self.api_key, self.engine_id, query)
        self.search_thread.finished.connect(
            lambda results, q: self.handle_search_results(results, q)
        )
        self.search_thread.start()

    def handle_search_results(self, results, query):
        if 'items' in results:
            #reversing items
            # results['items'] = results['items'][::-1]
            print('results', results['items'])
            self.update_verse_tracker(results['items'], query)

    #function to add saved verses to the saved pane
    def savedVerses(self, title, body):
        current_size = len(self.verse_widgets)
        link = os.path.join(os.path.dirname(__file__), '../ui/saved.ui')
        saved = loadUi(link)
        saved.title.setText(title)
        saved.body.setText(body)
        saved.delete_2.clicked.connect(lambda checked=False: self.delete(saved))
        #add one to len latter
        
        #adding action listener to title and body
        def mouse_click(event, verse_key):
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))

           
        # Bind `verse_key` explicitly
        saved.title.mousePressEvent = partial(mouse_click, verse_key=title)
        saved.body.mousePressEvent = partial(mouse_click, verse_key=title)

        self.searchPane.insertWidget(current_size+1, saved) #index 0 is the label widget
        if self.pop:
            return
        self.add_saved_verse(title, body)
        
    #deleting function for saved verses
    def delete(self, saved):
        print('deleting', saved.title.text())
        #finding the saved pane and deleting it
        index = self.searchPane.indexOf(saved)
        if index != -1:  # Widget found
            item = self.searchPane.takeAt(index)
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
        #using simulating the automata
        for index, action in enumerate(automata):
            print('prev action', prev_action)

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

        print("done")
    #change auto function
     #change auto function
    def changeAuto(self):
         # Lazy import
        from widgets.Welcome import Welcome
        
        # Clear current widgets from scroll area
        while self.scrollAreaWidgetContents.layout().count():
            item = self.scrollAreaWidgetContents.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Find the main window
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