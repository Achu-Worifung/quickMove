import os
from PyQt5.QtCore import Qt
from dotenv import load_dotenv
import math
from PyQt5.uic import loadUi
import httpx
from PyQt5.QtWidgets import (
    QDialog, QApplication, QLabel, QLineEdit, QMainWindow,
    QPushButton, QVBoxLayout, QComboBox, QWidget, QFrame, QCheckBox
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor
from collections import OrderedDict
import util.getReference as getReference
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, pyqtSlot
from functools import partial
from typing import Optional
import util.Simulate as Simulate
import util.savedVerses as savedVerses
from PyQt5.QtCore import QSettings
import re
import string
import json
from util.findVerseBox import findPrevDisplayedVerse
import time
import util.Message as msg
from util.util import resource_path
from util.transcription_class import TranscriptionWorker


class Tracker():
    def __init__(self, reference, result, priority):
        super().__init__()
        self.reference = reference
        self.result = result
        self.priority = priority

    def __repr__(self):
        return f"Tracker(reference='{self.reference}', priority={self.priority})"


class SearchThread(QThread):
    finished = pyqtSignal(list, str)

    def __init__(self, api_key: str, engine_id: str, query: str):
        super().__init__()
        self.api_key = api_key
        self.engine_id = engine_id
        self.query = query
        self._is_running = True

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
            with httpx.Client(timeout=5.0) as client:
                if not self._is_running:
                    self.finished.emit([], self.query)
                    return

                response = client.get(url, params=params, headers=header)

                if not self._is_running:
                    self.finished.emit([], self.query)
                    return

                response.raise_for_status()
                results = response.json()
                reversed_items = results.get("items", [])[::-1]

                if self._is_running:
                    self.finished.emit(reversed_items, self.query)

        except (httpx.RequestError, httpx.TimeoutException) as e:
            print(f"Search request failed: {e}")
            if self._is_running:
                self.finished.emit([], self.query)
        except Exception as e:
            print(f"Unexpected error in SearchThread: {e}")
            if self._is_running:
                self.finished.emit([], self.query)

    def stop(self):
        self._is_running = False


class locateVerseThread(QThread):
    finished = pyqtSignal(str)

    def run(self):
        self.finished.emit(findPrevDisplayedVerse())


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
        
        self.prev_verses = list()

        # basic UI defaults
        self.version.setCurrentIndex(0)
        self.data = data or {}
        self.name = (data or {}).get('name', 'unknown')
        self.verse_tracker = OrderedDict()
        self.search_thread: Optional[SearchThread] = None
        self.locate_box_thread: Optional[locateVerseThread] = None
        self.last_query_time = 0
        self.displayed_verse = []

        # transcription state
        self.transcription_thread: Optional[TranscriptionWorker] = None
        self.listening = False

        # load bible data for initial version
        url = os.path.join(os.path.dirname(__file__), f'../bibles/{self.version.currentText()}_bible.json')
        with open(url, 'r') as f:
            self.bible_data = json.load(f)

        # connect signals
        self.version.currentIndexChanged.connect(self.change_translation)
        
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse_step)
        self._pulse_state = 0

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
        self.refresh_search_widget()

        # focus out behaviour
        self.autoComplete_widget.lineedit.focusOutEvent = self.get_prev_verse
        self.prevVerse.clicked.connect(lambda checked=False: self.displayPrevVerse())
        self.record.clicked.connect(lambda checked=False: self.toggleListening())

    # ------------------------------------------------------------------
    # Transcription — inline, no separate window
    # ------------------------------------------------------------------
    def _set_button_state(self, state: str):
        """Update record button appearance based on transcription state."""
        self._pulse_timer.stop()

        styles = {
            "idle": (
                "QPushButton { background-color: #5F5E5A; border-radius: 8px; "
                "color: white; padding: 6px 14px; border: none; font-size: 13px; }"
            ),
            "processing": (
                "QPushButton { background-color: #185FA5; border-radius: 8px; "
                "color: white; padding: 6px 14px; border: none; font-size: 13px; }"
            ),
            "paused": (
                "QPushButton { background-color: #888780; border-radius: 8px; "
                "color: white; padding: 6px 14px; border: none; font-size: 13px; }"
            ),
        }

        labels = {
            "idle":       "Listen",
            "processing": "Processing...",
            "paused":     "Paused",
        }

        if state == "listening":
            # Start pulse timer — alternates between two blue shades
            self._pulse_state = 0
            self._pulse_timer.start(700)  # ms per step
            self.record.setText("Listening")
        else:
            self.record.setStyleSheet(styles.get(state, styles["idle"]))
            self.record.setText(labels.get(state, "Listen"))

    def _pulse_step(self):
        """Creates a smooth breathing effect using a sine wave."""
        # Increment state by a small amount for smoothness
        self._pulse_state += 0.15 
        
        # Calculate pulse value between 0 and 1
        pulse_val = (math.sin(self._pulse_state) + 1) / 2
        
        # Interpolate between a dark blue and a bright blue
        # Start: #185FA5 (24, 95, 165)
        # End:   #378ADD (55, 138, 221)
        r = int(24 + (55 - 24) * pulse_val)
        g = int(95 + (138 - 95) * pulse_val)
        b = int(165 + (221 - 165) * pulse_val)
        
        self.record.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r}, {g}, {b}); "
            f"border-radius: 8px; color: white; padding: 6px 14px; "
            f"border: none; font-size: 13px; }}"
        )
    def toggleListening(self):
        # Prevent double-clicking while processing
        if self.record.text() == "Processing...":
            return
            
        if self.listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        if self.transcription_thread and self.transcription_thread.isRunning():
            return

        self.listening = True
        self._pulse_state = 0  # Reset sine wave position
        self._set_button_state("listening") # Set style immediately

        self.transcription_thread = TranscriptionWorker(self, search_page=self.search_page)
        self.transcription_thread.autoSearchResults.connect(self.add_auto_search_results)
        self.transcription_thread.guitextReady.connect(self._update_transcription_text)
        self.transcription_thread.loadingStatus.connect(self._on_models_loaded)
        self.transcription_thread.statusUpdate.connect(self._on_status_update)
        self.transcription_thread.finished.connect(self._on_transcription_finished)

        self.history.setText("Loading...")
        QTimer.singleShot(100, self.transcription_thread.start)
        self.transcription_thread.start()

    def _stop_listening(self):
        if not self.listening:
            return
            
        self.listening = False # Flip immediately
        self._set_button_state("idle")
        
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.stop()
            # If the worker is stuck in a loop, wait briefly
            if not self.transcription_thread.wait(1000): 
                self.transcription_thread.terminate() # Nuclear option if it hangs
                
        self.history.setText("Stopped.")

    @pyqtSlot()
    def _on_models_loaded(self):
        self.history.setText("Listening...")
        print("Models loaded, ready to transcribe")

    @pyqtSlot(str)
    def _on_status_update(self, status: str):
        state_map = {
            "listening":  "listening",
            "processing": "processing",
            "paused":     "paused",
            "error":      "idle",
        }
        self._set_button_state(state_map.get(status, "idle"))
        
        # keep history label updated as before
        label_map = {
            "listening":  "Listening...",
            "processing": "Processing...",
            "paused":     "Paused",
            "error":      "Error — check API key",
        }
        self.history.setText(label_map.get(status, ""))

    @pyqtSlot(str)
    def _update_transcription_text(self, text: str):
        """Stream transcription into the dedicated transcription display field."""
        MAX_WORDS   = 200
        TRUNCATE_TO = 10

        current  = self.transcription_display.text()
        combined = (current + " " + text).strip()
        words    = combined.split()

        if len(words) > MAX_WORDS:
            combined = " ".join(words[-TRUNCATE_TO:])

        self.transcription_display.setText(combined)

    @pyqtSlot()
    def _on_transcription_finished(self):
        self.listening = False
        self._set_button_state("idle")
        self.history.setText("")

    def closeEvent(self, event):
        """Stop transcription cleanly when the widget closes."""
        self._stop_listening()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Existing methods — unchanged
    # ------------------------------------------------------------------

    def displayPrevVerse(self):
        # holder = findPrevDisplayedVerse()
        # print(f'prev verse {self.prev_verse_text} holder is {holder}')
        Simulate.present_prev_verse(self.name, self.data, self.prev_verse_text)
        # self.prev_verse_text = holder
        self.prev_verses.pop()
        self.updateHistoryLabel()

    def refresh_search_widget(self):
        from widgets.SearchBar import AutocompleteWidget
        self.autoComplete_widget = AutocompleteWidget(
            bar=self.search_bar_widget,
            width_widget=self.searchBarContainer
        )
        self.search_bar_widget.textChanged.connect(
            lambda text, d=self.data: self.handle_search(d, text)
        )
    def change_translation(self):
        url = os.path.join(os.path.dirname(__file__), f'../bibles/{self.version.currentText()}_bible.json')
        with open(url, 'r') as f:
            self.bible_data = json.load(f)
        self._refresh_saved_bodies()
        self._refresh_searchpane_bodies()

        current_query = self.autoComplete_widget.lineedit.text()
        if current_query and current_query.count(' ') >= 3:
            self.handle_search(self.data, current_query)

    def init_page(self):
        print("Initializing SearchWidget UI components")
        self.prevVerse          = self.search_page.findChild(QPushButton, 'prev_verse_3')
        self.version            = self.search_page.findChild(QComboBox, 'version_3')
        self.searchPane         = self.search_page.findChild(QVBoxLayout, 'searchPane_3')
        self.savedVerse_pane    = self.search_page.findChild(QVBoxLayout, 'saved_verse_pane')
        self.history            = self.search_page.findChild(QLineEdit, 'history')
        self.search_bar_widget  = self.search_page.findChild(QLineEdit, 'lineEdit_3')  # actual search bar
        self.transcription_display = self.search_page.findChild(QLineEdit, 'lineEdit')  # transcription display
        self.searchBarContainer = self.search_page.findChild(QWidget, 'widget_15')
        self.record             = self.search_page.findChild(QPushButton, 'listen')

        if self.version.count() == 0:
            self.version.addItems(['NLT', 'KJV', 'NIV', 'ASV', 'ESV'])

    def get_prev_verse(self, event):
        if self.locate_box_thread is None or not self.locate_box_thread.isRunning():
            self.locate_box_thread = locateVerseThread()
            self.locate_box_thread.finished.connect(self.handle_locate_results)
            self.locate_box_thread.start()
            super(QDialog, self).focusOutEvent(event)

    def handle_locate_results(self, result):
        result = re.sub(r'@', '', result).strip()
        result = ''.join(char for char in result if char in string.printable)
        new_result = re.sub(r'\).*$', '', result).strip()
        self.settings.setValue('displayed_reference', new_result)
        self.prev_verse_text = new_result
        self.updateHistoryLabel()

    def updateHistoryLabel(self):
        self.history.setText(f'Press Prev verse to Return to: {self.prev_verses[-1] if self.prev_verses else ""}')

    def clear_layout(self, layout):
        if layout is None:
            return
        widgets_to_delete = []
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widgets_to_delete.append(widget)
        for w in widgets_to_delete:
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception as e:
                print(f"Error deleting widget: {e}")

    def pop_saved_verse(self):
        get_saved_verse = savedVerses.getSavedVerses()
        if not get_saved_verse:
            self.savedVerse = {'savedVerses': []}
            return

        verses = get_saved_verse.get('savedVerses', [])
        self.savedVerse = {'savedVerses': []}
        self.clear_layout(self.savedVerse_pane)
        self.saved_widgets = []

        seen = set()
        for verse in verses:
            title = verse.get('title')
            if not title or title in seen:
                continue
            seen.add(title)

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

            self.savedVerse['savedVerses'].append({'title': title})
            self._create_saved_widget(title, body, persist=False)

    def delete_saved_verse(self, title):
        data = self.savedVerse.get('savedVerses', [])
        self.savedVerse['savedVerses'] = [v for v in data if v.get('title') != title]
        savedVerses.saveVerse(self.savedVerse)

    def add_saved_verse(self, title):
        data = self.savedVerse.setdefault('savedVerses', [])
        titles = {v.get('title') for v in data if v.get('title')}
        if title in titles:
            return
        data.insert(0, {'title': title})
        self.savedVerse['savedVerses'] = data
        savedVerses.saveVerse(self.savedVerse)

    def _create_saved_widget(self, title, body, persist=True):
        link = os.path.join(os.path.dirname(__file__), '../ui/saved.ui')
        saved = loadUi(link)
        saved.title.setText(title)
        saved.body.setText(body)
        saved.body.setStyleSheet("""
            text-align: left;
            font-size: 14px;
            line-height: 1.2em;
            max-height: 3.6em;
        """)
        saved.delete_2.clicked.connect(lambda checked=False: self.delete(saved))

        def mouse_click(event, verse_key):
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))

        saved.title.mousePressEvent = partial(mouse_click, verse_key=title)
        saved.body.mousePressEvent  = partial(mouse_click, verse_key=title)

        self.savedVerse_pane.insertWidget(0, saved)
        self.saved_widgets.insert(0, saved)

        if persist:
            self.add_saved_verse(title)

    def savedVerses(self, title, body):
        existing_titles = {v.get('title') for v in self.savedVerse.get('savedVerses', [])}
        if title in existing_titles:
            print(f"Already saved: {title}")
            return
        self._create_saved_widget(title, body, persist=True)

    def normalize_results(self, results):
        ref_counts = {}
        if not isinstance(results, list) or not all(isinstance(item, dict) and 'title' in item for item in results):
            raise ValueError("Expected results to be a list of dictionaries with a 'title' key.")
        for item in results:
            reference = getReference.getReference(item['title'].lower())
            if not reference:
                continue
            ref_counts[reference] = ref_counts.get(reference, 0) + 1
        return sorted(ref_counts.items(), key=lambda x: x[1])

    def update_verse_tracker(self, new_results, query):
        self.clear_layout(self.searchPane)
        self.old_widget.clear()
        self.displayed_verse.clear()
        print('current contents', self.searchPane.count())

        sorted_results = self.normalize_results(new_results)
        self.searchPane.update()
        for reference in sorted_results:
            print('here is the result in update', reference)
            self.displayed_verse.append(reference)
            self.add_verse_widget(query=query, reference=reference[0], result=None)

    def remove_widget(self, widget):
        if not widget:
            return
        self.searchPane.removeWidget(widget)
        widget.setParent(None)
        widget.deleteLater()

    def add_auto_search_results(self, results, query, confidence=None, max_results=10):
        results.sort(key=lambda x: x.get('score', 0), reverse=False)

        count = 0
        for result in results:
            if count >= max_results:
                break

            reference = (
                result.get('book', '').lower() + ' '
                + str(result.get('chapter', '')) + ':'
                + str(result.get('verse', ''))
            )

            if reference in self.displayed_verse:
                index = self.displayed_verse.index(reference)
                self.displayed_verse.pop(index)
                widget_to_remove = self.old_widget.pop(index)
                self.remove_widget(widget_to_remove)

            while len(self.displayed_verse) > 15:
                self.displayed_verse.pop(0)
                widget_to_remove = self.old_widget.pop(0)
                self.remove_widget(widget_to_remove)

            self.displayed_verse.append(reference)
            self.add_verse_widget(query, reference=reference, result=result, confidence=result['score'])
            count += 1

    def normalize_book(self, book: str) -> str:
        if not book:
            return book
        normalized_book, _, _ = getReference.parseReference(f"{book} 1:1")
        if not normalized_book:
            print(f"Warning: Normalization failed for '{book}'. Falling back to capitalize.")
            parts = ' '.join(book.strip().split()).split(' ')
            return ' '.join(p if p.isdigit() else p.capitalize() for p in parts)
        return normalized_book

    def add_verse_widget(self, query, reference, result=None, confidence=None):
        translation = self.version.currentText()
        link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
        single_result = loadUi(link)

        print('adding verse widget for reference:', reference)
        single_result.title.setText(reference.split('(')[0].strip())

        book, chapter, verse = getReference.parseReference(reference)
        body = "Verse not found in this translation."

        if book:
            normalized_book = self.normalize_book(book)
            parts = normalized_book.split(' ')
            if len(parts) > 1:
                normalized_book = parts[0] + ' ' + parts[1].capitalize()
            body = getReference.boldedText(
                self.bible_data.get(normalized_book, {}).get(str(chapter), {}).get(str(verse), "Verse not found in this translation."),
                query
            )
            body = body or "Verse not found in this translation."
            single_result.body.setText(body)
            single_result.body.setStyleSheet("""
                text-align: left;
                font-size: 14px;
                line-height: 1.2em;
                max-height: 3.6em;
            """)
        else:
            single_result.body.setText("Error parsing reference.")

        if confidence:
            single_result.confidence.setText(f'{confidence}')
        else:
            single_result.confidence.setStyleSheet('color: transparent;')

        single_result.save.clicked.connect(lambda checked=False, t=reference, b=body: self.savedVerses(t, b))
        self.searchPane.insertWidget(0, single_result)
        self.old_widget.append(single_result)

        def mouse_click(event, verse_key):
            QTimer.singleShot(0, lambda: self.present(verse_key, self.data))
            self.history.setText(f'Prev Displayed Verse: {verse_key}')

        single_result.title.mousePressEvent = partial(mouse_click, verse_key=reference)
        single_result.body.mousePressEvent  = partial(mouse_click, verse_key=reference)

    def handle_search(self, data=None, query=None):
        if query == "":
            self.double_search = True
            self.displayed_verse.clear()
            self.clear_layout(self.searchPane)
            self.old_widget.clear()
            self.verse_tracker = OrderedDict()

            if self.search_thread and self.search_thread.isRunning():
                self.search_thread.stop()
                self.search_thread.wait()

            self.search_thread = None
            self.last_query_time = 0
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

    def present(self, title, automata):
        self.settings.setValue("next_verse", title)
        self.prev_verse_text = findPrevDisplayedVerse()
        self.prev_verses.append(self.prev_verse_text)
        for action in automata.get('actions', []):
            if action['action'] == 'click':
                Simulate.simClick(action['location'][0], action['location'][1], action['button'])
            elif action['action'] == 'paste':
                Simulate.simPaste(title, True)
            elif action['action'] == 'select all':
                Simulate.simSelectAll(True)
                time.sleep(0.1)
        self.updateHistoryLabel()

    def _refresh_saved_bodies(self):
        for saved_widget in list(self.saved_widgets):
            try:
                title = saved_widget.title.text()
                book, chapter, verse_num = getReference.parseReference(title)
                if book:
                    normalized_book = self.normalize_book(book)
                    body = self.bible_data.get(normalized_book, {}).get(str(chapter), {}).get(str(verse_num), saved_widget.body.text())
                    saved_widget.body.setText(body)
            except Exception:
                continue

    def _refresh_searchpane_bodies(self):
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
                if book:
                    body = self.bible_data.get(book, {}).get(str(chapter), {}).get(str(verse_num), None)
                    if body:
                        current_query = self.autoComplete_widget.lineedit.text()
                        w.body.setText(getReference.boldedText(body, current_query))
            except Exception:
                pass