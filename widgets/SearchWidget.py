import os
from PyQt5.QtCore import Qt
from dotenv import load_dotenv
from PyQt5.uic import loadUi
import httpx
from PyQt5.QtWidgets import QDialog, QApplication, QLabel
from collections import OrderedDict
import util.getReference as getReference
from widgets.SearchBar import AutocompleteWidget
from PyQt5.QtCore import QThread, pyqtSignal, QMimeData, QTimer, pyqtSlot
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
import util.pyaudioandWhisper as transcriber
# from util import soundwave  # --- DISABLED TO PREVENT CRASH ---
from PyQt5.QtWidgets import QSizePolicy
import json
from util.findVerseBox import findPrevDisplayedVerse
import util.Message as msg
from util.util import resource_path


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
    """
    Runs a Google Custom Search API query in a separate thread.
    Includes a graceful stop mechanism.
    """
    finished = pyqtSignal(list, str)

    def __init__(self, api_key: str, engine_id: str, query: str):
        super().__init__()
        self.api_key = api_key
        self.engine_id = engine_id
        self.query = query
        self._is_running = True # Flag for graceful stop

    def run(self):
        url = 'https://customsearch.googleapis.com/customsearch/v1?'
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": self.query + ' KJV',
            "safe": "active",
            'hl': 'en',
            'num': 10,
        }
        header = {"User-Agent": "Mozilla/5.0"}

        try:
            # Use a client with a timeout so the thread doesn't block indefinitely
            with httpx.Client(timeout=5.0) as client:
                # Check if we were stopped before the request
                if not self._is_running:
                    self.finished.emit([], self.query)
                    return
                
                response = client.get(url, params=params, headers=header)

                # Check if we were stopped while waiting for the request
                if not self._is_running:
                    self.finished.emit([], self.query)
                    return

                response.raise_for_status()
                results = response.json()
                reversed_items = results.get("items", [])[::-1]

                # Final check before emitting
                if self._is_running:
                    self.finished.emit(reversed_items, self.query)

        except (httpx.RequestError, httpx.TimeoutException) as e:
            print(f"Search request failed: {e}")
            if self._is_running:
                self.finished.emit([], self.query) # Emit empty on error
        except Exception as e:
            print(f"An unexpected error occurred in SearchThread: {e}")
            if self._is_running:
                self.finished.emit([], self.query) # Emit empty on error

    def stop(self):
        """Signals the thread to stop gracefully."""
        self._is_running = False


class locateVerseThread(QThread):
    finished = pyqtSignal(str)

    def run(self):
        import util.findVerseBox as findVerseBox
        self.finished.emit(findVerseBox.findPrevDisplayedVerse())



class SearchWidget(QDialog):
    def __init__(self, search_page, data=None, index=None):
        super().__init__()
        print("SearchWidget init started")
        self.search_page = search_page
        self.init_page()

        # local state
        self.old_widget = []
        self.saved_widgets = []
        self.savedVerse = {'savedVerses': []}
        self.Tracker = []
        self.pop = True

        # basic UI defaults
        self.version.setCurrentIndex(0)
        self.data = data or {}
        self.name = (data or {}).get('name', 'unknown')
        self.verse_tracker = OrderedDict()
        # self.verse_widgets: Dict[str, QDialog] = {} # <-- Removed, was unused
        self.search_thread: Optional[SearchThread] = None
        self.locate_box_thread: Optional[locateVerseThread] = None
        self.last_query_time = 0
        self.displayed_verse = []

        # load bible data for initial version
        url = os.path.join(os.path.dirname(__file__), f'../bibles/{self.version.currentText()}_bible.json')
        with open(url, 'r') as f:
            self.bible_data = json.load(f)

        # connect signals
        self.version.currentIndexChanged.connect(self.change_translation)

        # qsettings
        self.settings = QSettings("MyApp", "AutomataSimulator")
        self.basedir = self.settings.value("basedir")

        # load env keys
        if self.basedir:
            load_dotenv(self.basedir + '/.env')
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('SEARCH_ENGINE_ID')

        # ensure saved pane cleared and populate from disk
        self.clear_layout(self.savedVerse_pane)
        self.saved_widgets = []
        self.savedVerse = {'savedVerses': []}
        self.pop = True
        self.pop_saved_verse()
        self.pop = False
        self.double_search = True

        # autocomplete
        from widgets.SearchBar import AutocompleteWidget
        self.autoComplete_widget = AutocompleteWidget(bar=self.search_bar[0], width_widget=self.searchBarContainer)
        self.search_bar[0].textChanged.connect(
            lambda text, d=self.data: self.handle_search(d, text)
        )

        # listening window
        self.listening_window = None
        self._creating_listening_window = False

        # focus out behaviour
        self.autoComplete_widget.lineedit.focusOutEvent = self.get_prev_verse
        self.prevVerse.clicked.connect(lambda checked=False: QTimer.singleShot(0, lambda: Simulate.present_prev_verse(self.name, self.data)))
        self.record.clicked.connect(lambda checked=False: self.startWhisper())


    def startWhisper(self):
        """Create/show listening window once; protected against re-entrant calls."""
        if self.listening_window is not None and self.listening_window.isVisible():
            return

        if self._creating_listening_window:
            return
        model_size = (QSettings("MyApp", "AutomataSimulator").value('model') or 'tiny').lower()
        model_path = resource_path(os.path.join(f'./models/{model_size}'))
        if not os.path.exists(model_path) or not os.listdir(model_path):
            msg.warningBox(self, "Model Missing", f"The Whisper model '{model_size}' is not found locally. Please ensure it is downloaded properly.")
            return
        print("Model found locally, proceeding...")
        self._creating_listening_window = True
        try:
            parent = self.search_page
            self.listening_window = WhisperWindow(parent, search_widget=self)
            self.listening_window.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            self.listening_window.show()
        finally:
            self._creating_listening_window = False


    def add_auto_search_results(self, results, query, confidence = None):
        print('here is the score', confidence)
        for result in results[::-1]: # reversing again here
            reference = getReference.getReference(result['title'])
            if not reference:
                continue
            
            # Check if this verse is already in our displayed list
            if reference in self.displayed_verse:
                print(f"Skipping duplicate auto-search: {reference}")
                continue # Skip this duplicate result
            else:
                self.displayed_verse.append(reference)
                print('appended to verse', self.displayed_verse[-1])
                # Only add the widget if it's not a duplicate
                self.add_verse_widget(query, result, reference, confidence=confidence)


    def change_translation(self):
        url = os.path.join(os.path.dirname(__file__), f'../bibles/{self.version.currentText()}_bible.json')
        with open(url, 'r') as f:
            self.bible_data = json.load(f)
        # refresh saved and search results bodies
        self._refresh_saved_bodies()
        self._refresh_searchpane_bodies()

        # re-run search if long query present
        current_query = self.autoComplete_widget.lineedit.text()
        if current_query and current_query.count(' ') >= 3:
            self.handle_search(self.data, current_query)


    def init_page(self):
        print("Initializing SearchWidget UI components")
        self.prevVerse = self.search_page.findChild(QPushButton, 'prev_verse_3')
        self.version = self.search_page.findChild(QComboBox, 'version_3')
        self.searchPane = self.search_page.findChild(QVBoxLayout, 'searchPane_3')
        self.savedVerse_pane = self.search_page.findChild(QVBoxLayout, 'saved_verse_pane')
        self.search_bar = self.search_page.findChildren(QLineEdit)
        self.history = self.search_page.findChild(QLineEdit, 'history')

        self.searchBarContainer = self.search_page.findChild(QWidget, 'widget_15')
        self.record = self.search_page.findChild(QPushButton, 'listen')
        # ensure version items exist once
        if self.version.count() == 0:
            self.version.addItems(['KJV', 'NIV', 'ESV', 'ASV', 'NLT'])

        print('listen button', self.record)


    def get_prev_verse(self, event):
        # Ensure the thread is only started once
        if self.locate_box_thread is None or not self.locate_box_thread.isRunning():
            self.locate_box_thread = locateVerseThread()
            self.locate_box_thread.finished.connect(self.handle_locate_results)
            self.locate_box_thread.start()
            super(QDialog, self).focusOutEvent(event) # Corrected super call


    def handle_locate_results(self, result):
        # Remove '@' from the result and strip extra spaces
        result = re.sub(r'@', '', result).strip()
        # Remove all non-printable characters from the result
        result = ''.join(char for char in result if char in string.printable)
        # removing the Bible reference using regex
        new_result = re.sub(r'\).*$', '', result).strip()
        self.settings.setValue('displayed_reference', new_result)
        self.updateHistoryLabel()


    def updateHistoryLabel(self):
        original = self.settings.value('copied_reference')
        displayed = self.settings.value('displayed_reference')
        self.history.setText(f'Original Reference: {original} Currently Displayed Verse: {displayed}')


    def prevVerse(self):
        print('data', self.data)
        Simulate.present_prev_verse(self.name)
        self.updateHistoryLabel()


    #
    # ------------------ NEW / FIX HELPERS BELOW ------------------
    #

    def clear_layout(self, layout):
        """Remove and delete all widget children of a QLayout."""
        if layout is None:
            return
        
        # Collect widgets to be deleted
        widgets_to_delete = []
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
                
            widget = item.widget()
            if widget is not None:
                widgets_to_delete.append(widget)

        # Delete all collected widgets
        for w in widgets_to_delete:
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception as e:
                print(f"Error deleting widget: {e}")
                pass


    def pop_saved_verse(self):
        """Load saved list from disk and create UI widgets once (no duplication)."""
        get_saved_verse = savedVerses.getSavedVerses()
        if not get_saved_verse:
            self.savedVerse = {'savedVerses': []}
            return

        verses = get_saved_verse.get('savedVerses', [])
        # Reset local storage to avoid duplicating on subsequent calls
        self.savedVerse = {'savedVerses': []}

        # Clear UI first (fix duplication when re-entering)
        self.clear_layout(self.savedVerse_pane)
        self.saved_widgets = []

        seen = set()
        for verse in verses:
            title = verse.get('title')
            if not title:
                continue
            if title in seen:
                continue
            seen.add(title)
            # lookup body using current bible_data; fallback to stored body if present
            stored_body = verse.get('body')
            book, chapter, verse_num = None, None, None
            try:
                book, chapter, verse_num = getReference.parseReference(title)
            except Exception:
                pass

            if book and chapter and verse_num:
                normalized_book = self.normalize_book(book)
                body = self.bible_data.get(normalized_book, {}).get(str(chapter), {}).get(str(verse_num), stored_body or "Verse not found.")
            else:
                body = stored_body or "Verse not found."

            # register in in-memory savedVerse structure but do NOT persist again
            self.savedVerse['savedVerses'].append({'title': title})
            # create UI widget but don't persist to disk (persist=False)
            self._create_saved_widget(title, body, persist=False)


    def delete_saved_verse(self, title):
        data = self.savedVerse.get('savedVerses', [])
        new_list = [verse for verse in data if verse.get('title') != title]
        self.savedVerse['savedVerses'] = new_list
        savedVerses.saveVerse(self.savedVerse)


    def add_saved_verse(self, title):
        """Persist a saved verse reference and update file storage (title-only)."""
        data = self.savedVerse.setdefault('savedVerses', [])
        titles = {v.get('title') for v in data if v.get('title')}
        if title in titles:
            return  # already saved
        data.insert(0, {'title': title})
        self.savedVerse['savedVerses'] = data
        savedVerses.saveVerse(self.savedVerse)


    def _create_saved_widget(self, title, body, persist=True):
        """Create the saved.ui widget and insert into the saved pane.
            persist=False prevents writing to disk (used when populating from disk)."""
        link = os.path.join(os.path.dirname(__file__), '../ui/saved.ui')
        saved = loadUi(link)
        saved.title.setText(title)
        saved.body.setText(body)
        saved.delete_2.clicked.connect(lambda checked=False: self.delete(saved))

        def mouse_click(event, verse_key):
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))

        saved.title.mousePressEvent = partial(mouse_click, verse_key=title)
        saved.body.mousePressEvent = partial(mouse_click, verse_key=title)

        # Insert at top
        self.savedVerse_pane.insertWidget(0, saved)
        self.saved_widgets.insert(0, saved)

        if persist:
            # persist only when user explicitly saves (not when populating UI)
            self.add_saved_verse(title)


    def savedVerses(self, title, body):
        """Called when user clicks "save" on a result - create widget and persist."""
        existing_titles = {v.get('title') for v in self.savedVerse.get('savedVerses', [])}
        if title in existing_titles:
            print(f"Already saved: {title}")
            return
        # Create widget and persist
        self._create_saved_widget(title, body, persist=True)




    def update_verse_tracker(self, new_results, query):
            widget_to_remove = []
            # clearing the layout of old results
            self.clear_layout(self.searchPane) # Use the robust clear_layout function

            self.old_widget.clear()
            self.displayed_verse.clear()  # Start with a fresh list
            print('current contents', self.searchPane.count())
            
            # --- START: Frequency Sort Logic ---
            ref_counts = {}
            for item in new_results:
                reference = getReference.getReference(item['title'])
                if not reference:
                    continue
                ref_counts[reference] = ref_counts.get(reference, 0) + 1

            def get_sort_key(result_item):
                ref = getReference.getReference(result_item['title'])
                return ref_counts.get(ref, 0)

            sorted_results = sorted(new_results, key=get_sort_key, reverse=True)
            print('Sorted results:', [r['title'] for r in sorted_results])
            # --- END: Frequency Sort Logic ---
            
            self.searchPane.update() 
            for result in sorted_results: # Use sorted list
                reference = getReference.getReference(result['title'])
                if not reference:
                    continue
                
                if reference in self.displayed_verse:
                    print(f"Skipping duplicate in new batch: {reference}")
                    continue 
                else:
                    self.displayed_verse.append(reference)
                    self.add_verse_widget(query, result, reference)


    def add_auto_search_results(self, results, query, confidence = None, max_results = 10):
            print('here is the score', confidence)
            
            # --- START: Frequency Sort Logic ---
            ref_counts = {}
            for item in results:
                reference = getReference.getReference(item['title'])
                if not reference:
                    continue
                ref_counts[reference] = ref_counts.get(reference, 0) + 1

            def get_sort_key(result_item):
                ref = getReference.getReference(result_item['title'])
                return ref_counts.get(ref, 0)
            
            sorted_results = sorted(results, key=get_sort_key, reverse=True)
            # --- END: Frequency Sort Logic ---

            count = 0
            for result in sorted_results: # Use sorted list
                if count >= max_results:
                    break
                
                reference = getReference.getReference(result['title'])
                if not reference:
                    continue

                if reference in self.displayed_verse:
                    print(f"Skipping duplicate auto-search: {reference}")
                    continue 
                
                self.displayed_verse.append(reference)
                print('appended to verse', self.displayed_verse[-1])
                self.add_verse_widget(query, result, reference, confidence=confidence)
                count += 1


    def callback(self, results, query, confidence = None):
        self.add_auto_search_results(results, query, confidence)


    def normalize_book(self, book: str) -> str:
        """
        Normalizes a book name to match the bible_data keys.
        This uses the canonical map from getReference.
        """
        # --- THIS IS THE NEW, ROBUST NORMALIZATION ---
        # It relies on your getReference.py file having the normalization functions
        if not book:
            return book
        
        # Use parseReference to get the *normalized* book name.
        # We pass a dummy chapter:verse to match the regex.
        normalized_book, _, _ = getReference.parseReference(f"{book} 1:1")
        
        # If normalization fails, fall back to simple capitalize
        if not normalized_book:
            print(f"Warning: Normalization failed for '{book}'. Falling back to capitalize.")
            parts = ' '.join(book.strip().split()).split(' ')
            normalized = []
            for p in parts:
                normalized.append(p if p.isdigit() else p.capitalize())
            return ' '.join(normalized)
            
        return normalized_book


    def add_verse_widget(self,query, result, reference, confidence = None):
        translation = self.version.currentText()
        link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
        single_result = loadUi(link)

        # --- THIS IS THE CORRECTED LOGIC ---
        # 1. 'reference' is the clean, canonical string (e.g., "Mark 12:30")
        single_result.title.setText(reference)

        # 2. Parse it to get the parts
        book, chapter, verse = getReference.parseReference(reference)

        if book:
            # 3. 'book' is now the canonical name (e.g., "Mark")
            # We no longer need self.normalize_book() here
            body = getReference.boldedText(
                self.bible_data.get(book, {}).get(str(chapter), {}).get(str(verse), "Verse not found in this translation."),
                query
            )
            single_result.body.setText(body)
        else:
            # Fallback if parsing fails (shouldn't happen if getReference worked)
            single_result.body.setText("Error parsing reference.")
        
        # --- (Rest of your function) ---
        if confidence:
            single_result.confidence.setText(f'{confidence:.2f}')
        else:
            single_result.confidence.setStyleSheet('color: transparent;')

        single_result.save.clicked.connect(lambda checked=False,t=reference, b=body: self.savedVerses(t, b))
        self.searchPane.insertWidget(0, single_result)
        self.old_widget.append(single_result)

        def mouse_click(event, verse_key):
            prev_verse = findPrevDisplayedVerse()
            print('prev_verse', prev_verse)
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))
            self.history.setText(f'Prev Displayed Verse: {prev_verse}')

        single_result.title.mousePressEvent = partial(mouse_click, verse_key=reference)
        single_result.body.mousePressEvent = partial(mouse_click, verse_key=reference)


    def handle_search(self, data=None, query=None):
        if query == "":
            self.double_search = True
            print('erasing query')

            self.displayed_verse.clear()
            # Use robust clear_layout
            self.clear_layout(self.searchPane) 

            self.old_widget.clear()
            self.verse_tracker = OrderedDict()
            
            # Stop any running search
            if self.search_thread and self.search_thread.isRunning():
                self.search_thread.stop()
                self.search_thread.wait()

            self.search_thread = None
            self.last_query_time = 0
            print('after erasing', self.searchPane.count())
            self.searchPane.update()
            return

        if not query or query.count(' ') < 3 or query[-1] != ' ':
            self.double_search = True
            return

        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.stop()
            self.search_thread.wait()

        self.search_thread = SearchThread(self.api_key, self.engine_id, query)
        self.search_thread.finished.connect(self.handle_search_results)
        self.search_thread.start()

    def handle_search_results(self, results, query):
        """Slot to receive results from SearchThread."""
        if results:
            self.update_verse_tracker(results, query)
        elif not results and self.searchPane.count() > 0 and self.double_search:
            print('no results')


    def delete(self, saved):
        index = self.savedVerse_pane.indexOf(saved)
        if index != -1:
            item = self.savedVerse_pane.takeAt(index)
            if item.widget():
                item.widget().deleteLater()
                self.delete_saved_verse(saved.title.text())
            
            if saved in self.saved_widgets:
                self.saved_widgets.remove(saved)
        pass

    def present(self, title, automata):
        self.settings.setValue("next_verse", title)
        prev_action = ''
        for action in automata.get('actions', []):
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
        self.updateHistoryLabel()


    #
    # -- helpers to refresh bodies when translation changes
    #
    def _refresh_saved_bodies(self):
        """Update displayed body texts of saved widgets to match current bible_data."""
        for saved_widget in list(self.saved_widgets):
            try:
                title = saved_widget.title.text()
                # parseReference *should* return the canonical book name
                book, chapter, verse_num = getReference.parseReference(title)
                if book:
                    body = self.bible_data.get(book, {}).get(str(chapter), {}).get(str(verse_num), saved_widget.body.text())
                    saved_widget.body.setText(body)
            except Exception:
                continue

    def _refresh_searchpane_bodies(self):
        """Update bodies in searchPane widgets to use current bible_data (if possible)."""
        for i in range(self.searchPane.count()):
            item = self.searchPane.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is None:
                continue
            try:
                reference = w.title.text()
                # parseReference *should* return the canonical book name
                book, chapter, verse_num = getReference.parseReference(reference)
                if book:
                    body = self.bible_data.get(book, {}).get(str(chapter), {}).get(str(verse_num), None)
                    if body:
                        # Re-apply bolding based on the *current* query
                        current_query = self.autoComplete_widget.lineedit.text()
                        w.body.setText(getReference.boldedText(body, current_query))
            except Exception:
                pass


#
# Listening UI + worker classes
#
class WhisperWindow(QFrame):
    def __init__(self, parent=None, search_widget=None):
        super().__init__(parent)
        

        link = os.path.join(os.path.dirname(__file__), '../ui/listening_window.ui')
        self.listening_window = loadUi(link, self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.close_button.clicked.connect(self.close)
        self.lineEdit = self.findChild(QLineEdit, 'lineEdit')
        self.record_btn = self.checkBox
        self.checkBox.setChecked(True)
        self.checkBox.clicked.connect(self.start_recording)
        self.search_page = parent
        self.search_widget = search_widget

        # --- THIS IS THE CRITICAL CRASH-FIX ---
        # 1. Create the worker, passing the lineEdit
        self.transcription_thread = TranscriptionWorker(
            self, 
            search_page=parent, 
            lineEdit=self.lineEdit # Pass the widget
        )
        
        # 2. Connect the auto-search signal
        self.transcription_thread.autoSearchResults.connect(self.search_widget.add_auto_search_results)
        
        # 3. Connect the thread-safe text signal to the GUI slot
        self.transcription_thread.guiTextReady.connect(self.update_transcription_text)

        # 4. Start the thread
        self.transcription_thread.start()
        # --- END CRASH-FIX BLOCK ---

        # --- SOUNDWAVE VISUALIZER DISABLED TO PREVENT CRASH ---
        # self.soundwave_label = soundwave.SoundWaveLabel(self)
        # original_label = self.findChild(QLabel, 'sound')
        # if original_label:
        #     geometry = original_label.geometry()
        #     parent_widget = original_label.parent()
        #     original_label.setParent(None)
        #     self.soundwave_label.setParent(parent_widget)
        #     self.soundwave_label.setGeometry(geometry)
        #     self.soundwave_label.setText("Not listening")
        #     self.soundwave_label.setStyleSheet('margin: 30px auto; color: white;')
        #     self.label = self.soundwave_label
        #     print("Soundwave label created and replaced original label")
        # else:
        #     print("Original 'sound' label not found!")

        # if self.checkBox.isChecked():
        #     print("Checkbox is checked, starting visualization automatically")
        #     self.soundwave_label.start_recording_visualization()
        # --- END SOUNDWAVE BLOCK ---

    @pyqtSlot(str) 
    def update_transcription_text(self, text):
        """
        This function is *guaranteed* to run on the main GUI thread.
        It is the ONLY safe place to call self.lineEdit.setText().
        """
        current_ui_text = self.lineEdit.text()
        self.lineEdit.setText((current_ui_text + " " + text).strip())

    def start_recording(self):
        print(f"Start recording called, checkbox checked: {self.record_btn.isChecked()}")
        if self.record_btn.isChecked():
            if not self.transcription_thread.isRunning():
                # Create new thread if old one finished
                self.transcription_thread = TranscriptionWorker(self, search_page=self.search_page, lineEdit=self.lineEdit)
                self.transcription_thread.autoSearchResults.connect(self.search_widget.add_auto_search_results)
                self.transcription_thread.guiTextReady.connect(self.update_transcription_text)
                self.transcription_thread.start()
            
            # --- SOUNDWAVE DISABLED ---
            # print("Starting soundwave visualization")
            # self.soundwave_label.start_recording_visualization()
        else:
            print("Stopping soundwave visualization")
            # Use graceful stop instead of terminate
            if self.transcription_thread.isRunning():
                self.transcription_thread.stop_transcription()
                if not self.transcription_thread.wait(5000):
                    print("Transcription didn't stop gracefully, forcing termination")
                    self.transcription_thread.terminate()
                    self.transcription_thread.wait()
            
            # --- SOUNDWAVE DISABLED ---
            # self.soundwave_label.stop_recording_visualization()

    def close(self):
        print("WhisperWindow closing")
        
        # --- SOUNDWAVE DISABLED ---
        # if hasattr(self, 'soundwave_label'):
        #     self.soundwave_label.stop_recording_visualization()
        
        # Graceful shutdown of transcription thread
        if hasattr(self, 'transcription_thread') and self.transcription_thread.isRunning():
            print("Requesting graceful transcription shutdown...")
            self.transcription_thread.stop_transcription()
            
            if not self.transcription_thread.wait(5000):
                print("Timeout waiting for graceful shutdown, forcing termination")
                self.transcription_thread.terminate()
                self.transcription_thread.wait()
            else:
                print("Transcription stopped gracefully")
        
        if self.search_widget:
            self.search_widget.listening_window = None
            print('Cleared listening_window reference in SearchWidget')
        
        super().close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()
        else:
            self.oldPos = None
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.oldPos = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.oldPos is None:
            return
        delta = event.globalPos() - self.oldPos
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

class TranscriptionWorker(QThread):
    finished = pyqtSignal()
    autoSearchResults = pyqtSignal(list, str, float)  # Emit results, query, and confidence
    guiTextReady = pyqtSignal(str) # The safe signal for the GUI

    def __init__(self, parent=None, search_page=None, lineEdit=None):
            super().__init__(parent)
            self.record_page = parent
            self.search_page = search_page
            self.lineEdit = lineEdit # Pass lineEdit to run_transcription
            self.controller = transcriber.TranscriptionController()

    def run(self):
        transcriber.run_transcription(
            recording_page=self.record_page,
            search_Page=self.search_page,
            lineEdit=self.lineEdit, # Pass it (though it's no longer used for .setText)
            worker_thread=self,     # Pass self (this QThread)
            controller=self.controller
        )
        self.finished.emit()
    
    def stop_transcription(self):
        print("Stopping transcription gracefully...")
        self.controller.stop()