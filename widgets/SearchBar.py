import xml.etree.ElementTree as ET
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork

from PyQt5.QtWidgets import *


class StyledPopup(QtWidgets.QListView):
    def __init__(self, search_bar, width_widget=None, parent=None):
        super(StyledPopup, self).__init__(parent)
        
        self.search_bar = search_bar  # Store reference to the actual search bar
        self.width_widget = width_widget or search_bar  # Widget to match width from
        
        self.setWindowFlags(QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setUniformItemSizes(True)

        self.setStyleSheet("""
            QListView {
                border: 1px solid #ccc;
                background-color: #dcdcdc;
                color: black;
                outline: 0;
                padding: 5px;
                font-size: 15px;
                font: MS Shell Dlg 2;
                border-radius: 8px;
            }
            QListView::item {
                padding: 6px;
                border-radius: 5px;
            }
            QListView::item:selected {
                background-color: #e6e6e6;
                color: black;
            }
            QListView::item:hover {
                background-color: #f0f0f0;
            }
        """)

    ## ========================================================================
    ##  THE FIX: SIMPLIFIED LOGIC
    ##  Align everything (Width, X, and Y) to the search_bar (QLineEdit)
    ## ========================================================================
    def showEvent(self, event):
        """
        Align popup width and position to the width_widget (container with eye icon)
        """
        
        # 1. Use width_widget for WIDTH
        self.setFixedWidth(self.width_widget.width())

        # 2. Calculate height
        row_h = self.sizeHintForRow(0) if self.model().rowCount() else 30
        max_h = row_h * min(6, self.model().rowCount() or 1) + 10
        self.setFixedHeight(max_h)

        # 3. Use width_widget for POSITION (X and Y) instead of search_bar
        pos = self.width_widget.mapToGlobal(QtCore.QPoint(0, self.width_widget.height()))
        
        # 4. Check if it goes off the bottom of the available screen
        desktop = QtWidgets.QApplication.desktop()
        screen_rect = desktop.availableGeometry(self.width_widget) 
        
        if pos.y() + max_h > screen_rect.bottom():
            # If it goes off the bottom, move it *above* the widget
            pos = self.width_widget.mapToGlobal(QtCore.QPoint(0, -max_h))
            
        self.move(pos)
        
        super().showEvent(event)


class SuggestionModel(QtGui.QStandardItemModel):
    def __init__(self, parent=None):
        super(SuggestionModel, self).__init__(parent)
        self._manager = QtNetwork.QNetworkAccessManager(self)
        self._reply = None
        self.suggestions = set()
        self.max_suggestions = QtCore.QSettings().value('suggestion_length', 3, type=int)

    @QtCore.pyqtSlot(str)
    def search(self, text):
        if self._reply is not None:
            self._reply.abort()
        if text:
            r = self.create_request(text)
            self._reply = self._manager.get(r)
            self._reply.finished.connect(self.on_finished)
        else:
            self.clear()
            self.suggestions.clear()

    def create_request(self, text):
        url = QtCore.QUrl("http://toolbarqueries.google.com/complete/search")
        query = QtCore.QUrlQuery()
        query.addQueryItem("q", text)
        query.addQueryItem("output", "toolbar")
        query.addQueryItem("hl", "en")
        url.setQuery(query)
        request = QtNetwork.QNetworkRequest(url)
        return request

    @QtCore.pyqtSlot()
    def on_finished(self):
        """
         Handle the finished signal from the network reply.
         for the suggestions"""
        reply = self.sender()
        if reply.error() == QtNetwork.QNetworkReply.NoError:
            try:
                content = reply.readAll().data()
                suggestions = ET.fromstring(content)
                self.clear()
                self.suggestions.clear()
                print('here are the suggestions', suggestions)
                
                for data in suggestions.iter("suggestion"):
                    suggestion = data.attrib["data"]
                    if suggestion not in self.suggestions:
                        self.suggestions.add(suggestion)
                        item = QtGui.QStandardItem(suggestion)
                        item.setFont(QtGui.QFont("MS Shell Dlg 2", 12))  
                        self.appendRow(item)
                        if self.rowCount() >= self.max_suggestions:
                            break
            except ET.ParseError:
                print("Error parsing suggestions XML")
            except Exception as e:
                print(f"Error processing suggestions: {str(e)}")
        
        reply.deleteLater()
        self._reply = None


class Completer(QtWidgets.QCompleter):
    def __init__(self, search_bar, width_widget=None, parent=None, **kwargs):
        super(Completer, self).__init__(parent, **kwargs)
        self.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)

        # pass search_bar and width_widget to popup
        self.popup = StyledPopup(search_bar, width_widget, parent)
        self.setPopup(self.popup)

    def splitPath(self, path):
        self.model().search(path)
        return []


class AutocompleteWidget(QtWidgets.QWidget):
    def __init__(self, bar=None, width_widget=None, parent=None):
        super(AutocompleteWidget, self).__init__(parent)
        
        print('bar', type(bar))
        self.bar = bar
        self.width_widget = width_widget  # Store the widget to match width from
        
        self._model = SuggestionModel(self)
        
        self.lineedit = self.bar if bar else QtWidgets.QLineEdit(self)
        self.lineedit.setPlaceholderText("Search...")

        if not bar:
            layout = QtWidgets.QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.lineedit)
            self.setLayout(layout)
        
        active_width_widget = self.width_widget or self
        
        completer = Completer(self.lineedit, active_width_widget, self, caseSensitivity=QtCore.Qt.CaseInsensitive)
        completer.setModel(self._model)
        
        self.lineedit.setCompleter(completer)
        self.lineedit.returnPressed.connect(self.handle_enter)
        

    def handle_enter(self):
        if self._model.rowCount() > 0:
            top_item = self.model().item(0)
            if top_item:
                self.lineedit.setText(top_item.text() + ' ')
                return


class AnotherWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(AnotherWidget, self).__init__(parent)
        
        self.autocomplete_widget = AutocompleteWidget(parent=self)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.autocomplete_widget)
        
        other_label = QtWidgets.QLabel("This is another widget", self)
        layout.addWidget(other_label)
        
        self.setLayout(layout)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = AnotherWidget()
    w.resize(400, 200)
    w.show()
    sys.exit(app.exec_())