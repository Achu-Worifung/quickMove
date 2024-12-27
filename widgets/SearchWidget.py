import os
from dotenv import load_dotenv
from PyQt5.uic import loadUi
import httpx
from PyQt5.QtWidgets import QDialog, QApplication
from collections import OrderedDict
import util.getReference as getReference
from widgets.SearchBar import AutocompleteWidget
from PyQt5.QtCore import QThread, pyqtSignal
import asyncio
from functools import partial
from typing import Dict, Optional
import util.Simulate as Simulate

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
            "q": f'{self.query} ',
            "num": 10
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
        self.version.setCurrentIndex(0)
        self.data = data
        # print('data', data)
        self.verse_tracker = OrderedDict()
        self.verse_widgets: Dict[str, QDialog] = {}  # Store widget references
        self.search_thread: Optional[SearchThread] = None
        self.last_query_time = 0
        
        autoComplete_widget = AutocompleteWidget(self)
        self.horizontalLayout.addWidget(autoComplete_widget)
        autoComplete_widget.lineedit.textChanged.connect(
            lambda text, d=self.data: self.handle_search(d, text)
        )

        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('SEARCH_ENGINE_ID')

    def update_verse_tracker(self, new_results, query):
        for result in new_results[:10]:
            title_array = getReference.getReference(result['title'])
            if not title_array:
                continue
            
            verse_key = title_array[0]
            if verse_key in self.verse_tracker:
                self.verse_tracker.move_to_end(verse_key)
                self.verse_tracker[verse_key]['priority'] += 1
                # Update existing widget
                if verse_key in self.verse_widgets:
                    body = getReference.boldedText(result['snippet'], query)
                    self.verse_widgets[verse_key].body.setText(body)
            else:
                if len(self.verse_tracker) < 10:
                    self.verse_tracker[verse_key] = {
                        'priority': 1,
                        'data': result,
                        'query': query
                    }
                    self.add_verse_widget(verse_key, result, query)
                else:
                    lowest_key = next(iter(self.verse_tracker))
                    if self.verse_tracker[lowest_key]['priority'] < 1:
                        # Remove lowest priority widget
                        if lowest_key in self.verse_widgets:
                            widget = self.verse_widgets.pop(lowest_key)
                            self.searchPane.removeWidget(widget)
                            widget.deleteLater()
                        
                        self.verse_tracker.popitem(last=False)
                        self.verse_tracker[verse_key] = {
                            'priority': 1,
                            'data': result,
                            'query': query
                        }
                        self.add_verse_widget(verse_key, result, query)

    def add_verse_widget(self, verse_key, result, query):
        link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
        single_result = loadUi(link)
        body = getReference.boldedText(result['snippet'], query)
        single_result.body.setText(body)
        single_result.title.setText(verse_key)
        print('verse_key', verse_key)
        
        single_result.save.clicked.connect(lambda checked=False,t=verse_key, b=body: self.savedVerses(t, b))
        self.searchPane.addWidget(single_result)
        self.verse_widgets[verse_key] = single_result
        def mouse_click(event):
            self.present(verse_key, self.data)
            print('the result was clicked')
        single_result.title.mousePressEvent = mouse_click
        single_result.body.mousePressEvent = mouse_click
        
        # self.searchPane.single_result.clicked.connect(self.present)  

    def handle_search(self, data=None, query=None):
        if query == "":
            while self.searchPane.count():
                widget = self.searchPane.takeAt(0).widget()
                if widget:
                    widget.deleteLater()
            self.verse_tracker = OrderedDict()
            self.verse_widgets: Dict[str, QDialog] = {}  # Store widget references
            self.search_thread: Optional[SearchThread] = None
            self.last_query_time = 0
            return
        print('here is the query', query)

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
            self.update_verse_tracker(results['items'], query)

    def savedVerses(self, title, body):
        print('title', title)
        print('body', body)
        link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
        saved = loadUi(link)
        saved.title.setText(title)
        saved.body.setText(body)
        # len = len(self.verse_widgets)
        #add one to len latter

        self.searchPane.insertWidget(len, saved) #index 0 is the label widget
        

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
        print("done")
 