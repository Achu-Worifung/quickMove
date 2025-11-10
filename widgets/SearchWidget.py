import os
from PyQt5.QtCore import Qt
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
import util.pyaudioandWhisper as transcriber
from util import soundwave
from PyQt5.QtWidgets import QSizePolicy
import json
from util.findVerseBox import findPrevDisplayedVerse


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
        url = 'https://customsearch.googleapis.com/customsearch/v1?'
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": self.query + ' KJV',
            "safe": "active",
            'hl': 'en',
            'num': 10,
        }
        header =  {"User-Agent": "Mozilla/5.0"}
        response = httpx.get(url, params=params, headers=header)
        response.raise_for_status()
        results = response.json()
        reversed_items = results.get("items", [])[::-1]
        self.finished.emit(reversed_items, self.query)


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
        self.verse_widgets: Dict[str, QDialog] = {}
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

        self._creating_listening_window = True
        try:
            parent = self.search_page
            self.listening_window = WhisperWindow(parent, search_widget=self)
            self.listening_window.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            self.listening_window.show()
        finally:
            self._creating_listening_window = False


    def add_auto_search_results(self, results, query):
       print("add_auto_search_results called on main thread.") # Debug print
       for result in results:
              reference = getReference.getReference(result['title'])
              if not reference:
                continue
              self.add_verse_widget(query, result, reference)


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
            super().focusOutEvent(event)


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
        widgets = []
        # iterate in reverse to be safe
        for i in reversed(range(layout.count())):
            item = layout.takeAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                widgets.append(w)
        for w in widgets:
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception:
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
                body = self.bible_data.get(book, {}).get(str(chapter), {}).get(str(verse_num), stored_body or "Verse not found.")
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
            return
        # Create widget and persist
        self._create_saved_widget(title, body, persist=True)


    #
    # ------------------ EXISTING SEARCH + RESULT LOGIC ------------------
    #

    def update_verse_tracker(self, new_results, query):
        widget_to_remove = []
        # clearing the layout of old results
        for i in range(self.searchPane.count()):
            widget_to_remove.append(self.searchPane.itemAt(i).widget())
        for widget in widget_to_remove:
            self.searchPane.removeWidget(widget)
            if widget:
                widget.deleteLater()
        self.old_widget.clear()
        self.displayed_verse.clear()
        print('current contedents', self.searchPane.count())
        # adding the new results to the layout
        self.searchPane.update()
        for result in new_results:
            reference = getReference.getReference(result['title'])
            if not reference:
                continue
            if reference in self.displayed_verse:
                index = self.displayed_verse.index(reference)
                remove_widget = self.searchPane.takeAt(index)
                if remove_widget.widget():
                    remove_widget.widget().deleteLater()
                    remove_widget.widget().setParent(None)
            else:
                self.displayed_verse.append(reference)
            self.add_verse_widget(query, result, reference)


    def add_auto_search_results(self, results, query, confidence = None):
        print('here is the score', confidence)
        for result in results[::-1]: # reversing again here
            reference = getReference.getReference(result['title'])
            if not reference:
                continue
            if reference in self.displayed_verse:
                index = self.displayed_verse.index(reference)
                remove_widget = self.searchPane.takeAt(index)
                if remove_widget.widget():
                    remove_widget.widget().deleteLater()
                    remove_widget.widget().setParent(None)
            else:
                self.displayed_verse.append(reference)
                print('appended to verse', self.displayed_verse[-1])
            self.add_verse_widget(query, result, reference, confidence=confidence)


    def callback(self, results, query, confidence = None):
        self.add_auto_search_results(results, query, confidence)


    def add_verse_widget(self,query, result, reference, confidence = None):
        translation = self.version.currentText()
        link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
        single_result = loadUi(link)

        book, chapter, verse = getReference.parseReference(reference)

        body = self.bible_data.get(book, {}).get(str(chapter), {}).get(str(verse), "Verse not found in this translation.")
        single_result.body.setText(body)
        single_result.title.setText(reference)
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
            widgets_to_remove = []
            for i in range(self.searchPane.count()):
                widget = self.searchPane.itemAt(i).widget()
                if widget:
                    widgets_to_remove.append(widget)

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

        if not query or query.count(' ') < 3 or query[-1] != ' ':
            self.double_search = True
            return

        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.terminate()
            self.search_thread.wait()

        self.search_thread = SearchThread(self.api_key, self.engine_id, query)
        self.search_thread.finished.connect(
            lambda results, q=query: self.handle_search_results(results, q)
        )
        self.search_thread.start()

    def handle_search_results(self, results, query):
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
                book, chapter, verse_num = getReference.parseReference(title)
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
                book, chapter, verse_num = getReference.parseReference(reference)
                body = self.bible_data.get(book, {}).get(str(chapter), {}).get(str(verse_num), None)
                if body:
                    w.body.setText(body)
            except Exception:
                pass


#
# Listening UI + worker classes (kept mostly as-is, small cleanups)
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

        # Start transcription in a separate thread
        self.transcription_thread = TranscriptionWorker(self, search_page=parent)
        self.transcription_thread.autoSearchResults.connect(self.search_widget.add_auto_search_results)
        self.transcription_thread.start()

        self.soundwave_label = soundwave.SoundWaveLabel(self)

        original_label = self.findChild(QLabel, 'sound')
        if original_label:
            geometry = original_label.geometry()
            parent_widget = original_label.parent()
            original_label.setParent(None)
            self.soundwave_label.setParent(parent_widget)
            self.soundwave_label.setGeometry(geometry)
            self.soundwave_label.setText("Not listening")
            self.soundwave_label.setStyleSheet('margin: 30px auto; color: white;')
            self.label = self.soundwave_label
            print("Soundwave label created and replaced original label")
        else:
            print("Original 'sound' label not found!")

        if self.checkBox.isChecked():
            print("Checkbox is checked, starting visualization automatically")
            self.soundwave_label.start_recording_visualization()

    def start_recording(self):
        print(f"Start recording called, checkbox checked: {self.record_btn.isChecked()}")
        if self.record_btn.isChecked():
            if not self.transcription_thread.isRunning():
                self.transcription_thread.start()
            print("Starting soundwave visualization")
            self.soundwave_label.start_recording_visualization()
        else:
            print("Stopping soundwave visualization")
            self.transcription_thread.terminate()
            self.soundwave_label.stop_recording_visualization()

    def close(self):
        print("WhisperWindow closing")
        if hasattr(self, 'soundwave_label'):
            self.soundwave_label.stop_recording_visualization()
        if self.transcription_thread.isRunning():
            self.transcription_thread.terminate()
            self.transcription_thread.wait()
            print('Terminated transcription thread')
        if self.search_widget:
            self.search_widget.listening_window = None
            self.record_btn.setChecked(False)
            
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

    def __init__(self, parent=None, search_page = None):
        super().__init__(parent)
        self.record_page = parent
        self.search_page = search_page
        self.lineEdit = self.record_page.lineEdit

    def run(self):
        transcriber.run_transcription(
            recording_page=self.record_page,
            search_Page=self.search_page,
            lineEdit=self.lineEdit,
            worker_thread=self
        )
        self.finished.emit()
