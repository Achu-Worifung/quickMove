import os
from dotenv import load_dotenv
from PyQt5.uic import loadUi
import httpx
from PyQt5.QtWidgets import QDialog, QApplication, QLabel
from collections import OrderedDict
import util.getReference as getReference
from widgets.SearchBar import AutocompleteWidget
from PyQt5.QtCore import QThread, pyqtSignal
import asyncio
from functools import partial
from typing import Dict, Optional
import util.Simulate as Simulate
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
import util.savedVerses as savedVerses


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
            "q": f'"{self.query}" site:biblehub.com',
            "allintitle": self.query,
            "allintext": self.query,

        }
        response = httpx.get(url, params=params)
        response.raise_for_status()
        self.finished.emit(response.json(), self.query)

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
        self.last_query_time = 0
        
        autoComplete_widget = AutocompleteWidget(self)
        autoComplete_widget.setStyleSheet('height: 50px; border-radius: 10px; font-size: 20px;')
        autoComplete_widget.lineedit.setPlaceholderText('   Search for a verse')
        self.horizontalLayout.addWidget(autoComplete_widget)
        autoComplete_widget.lineedit.textChanged.connect(
            lambda text, d=self.data: self.handle_search(d, text)
        )
         #adding action listerner to change auto button
        self.pushButton.clicked.connect(lambda checked=False: self.changeAuto())


        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('SEARCH_ENGINE_ID')

        self.pop_saved_verse()
        self.pop = False
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
        print('verse tracker', self.verse_tracker, '\n\n\n')
        print('verse widgets', self.verse_widgets, '\n\n\n')
        print('top ten results', new_results[:10])

        new_results = new_results[::-1]
        new_keys = []

        for result in new_results[:10]:
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

            
            # verse_key = title_array[0]
            # if verse_key in self.verse_tracker:
            #     self.verse_tracker.move_to_end(verse_key)
            #     self.verse_tracker[verse_key]['priority'] += 1
            #     # Update existing widget
            #     if verse_key in self.verse_widgets:
            #         body = getReference.boldedText(result['snippet'], query)
            #         self.verse_widgets[verse_key].body.setText(body)
            # else:
            #     if len(self.verse_tracker) < 10:
            #         self.verse_tracker[verse_key] = {
            #             'priority': 1,
            #             'data': result,
            #             'query': query
            #         }
            #         self.add_verse_widget(verse_key, result, query)
            #     else:
            #         lowest_key = next(iter(self.verse_tracker))
            #         if self.verse_tracker[lowest_key]['priority'] < 1:
            #              # Remove lowest priority widget
            #             if lowest_key in self.verse_widgets:
            #                 widget = self.verse_widgets.pop(lowest_key)
            #                 self.searchPane.removeWidget(widget)
            #                 widget.deleteLater()
                        
            #             self.verse_tracker.popitem(last=False)
            #             self.verse_tracker[verse_key] = {
            #                 'priority': 1,
            #                 'data': result,
            #                 'query': query
            #             }
            #             self.add_verse_widget(verse_key, result, query)

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
        def mouse_click(event):
            self.present(verse_key, self.data)
           
        single_result.title.mousePressEvent = mouse_click
        single_result.body.mousePressEvent = mouse_click
        
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

        if not query or query.count(' ') < 3:
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
            results['items'] = results['items'][::-1]
            # print('results', results)
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
        clipboard = QApplication.clipboard()
        clipboard.setText(title)
        print(title)
        print('copied to clipboard')
        print(automata)
        #using simulating the automata
        for action in automata:
            if action['action'] == 'click':
                x_coord = action['location'][0]
                y_coord = action['location'][1]
                button = action['button']
                Simulate.simClick(x_coord, y_coord, button, True)
            elif action['action'] == 'paste':
                Simulate.simPaste('v', True)
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