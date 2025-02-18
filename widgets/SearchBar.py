import xml.etree.ElementTree as ET
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork

from PyQt5.QtWidgets import *


class StyledPopup(QtWidgets.QListView):
    def __init__(self, parent=None):
        super(StyledPopup, self).__init__(parent)
        # Set custom properties for the popup
        self.setWindowFlags(QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setStyleSheet("""
            QListView {
                border: 1px solid #ccc;
                border-radius: 10px;
                background-color: white;
                outline: 0;
                padding: 5px;
            }
            QListView::item {
                padding: 5px;
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


class SuggestionModel(QtGui.QStandardItemModel):
    def __init__(self, parent=None):
        super(SuggestionModel, self).__init__(parent)
        self._manager = QtNetwork.QNetworkAccessManager(self)
        self._reply = None
        self.suggestions = set()

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
        reply = self.sender()
        if reply.error() == QtNetwork.QNetworkReply.NoError:
            try:
                content = reply.readAll().data()
                suggestions = ET.fromstring(content)
                self.clear()
                self.suggestions.clear()
                
                for data in suggestions.iter("suggestion"):
                    suggestion = data.attrib["data"]
                    if suggestion not in self.suggestions:
                        self.suggestions.add(suggestion)
                        item = QtGui.QStandardItem(suggestion)
                        item.setFont(QtGui.QFont("MS Shell Dlg 2", 12))  # Font and size
                        self.appendRow(item)
            except ET.ParseError:
                print("Error parsing suggestions XML")
            except Exception as e:
                print(f"Error processing suggestions: {str(e)}")
        
        reply.deleteLater()
        self._reply = None


class Completer(QtWidgets.QCompleter):
    def __init__(self, parent=None, **kwargs):
        super(Completer, self).__init__(parent, **kwargs)
        self.setCompletionMode(QtWidgets.QCompleter.UnfilteredPopupCompletion)
        
        # Set the custom popup
        self.popup = StyledPopup()
        self.setPopup(self.popup)

    def splitPath(self, path):
        self.model().search(path)
        return []


class AutocompleteWidget(QtWidgets.QWidget):
    def __init__(self, bar=None):
        super(AutocompleteWidget, self).__init__(bar)
        
        print('bar', type( bar))
        self.bar = bar
        
        self._model = SuggestionModel(self)
        completer = Completer(self, caseSensitivity=QtCore.Qt.CaseInsensitive)
        completer.setModel(self._model)
    
        
        self.lineedit = self.bar if bar else QtWidgets.QLineEdit(self)
        # self.lineedit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineedit.setCompleter(completer)
        
        # Set minimum height for the line edit
        # self.lineedit.setMinimumHeight(35)
        
        # Apply styling
        # self.lineedit.setStyleSheet("""
        #     QLineEdit {
        #         border-radius: 10px;
        #         font-size: 20px;
        #         padding: 0 10px;
        #         border: 1px solid #ccc;
        #         background-color: white;
        #     }
        #     QLineEdit:focus {
        #         border: 1px solid #4a90e2;
        #         outline: none;
        #     }
        # """)

        # layout = QtWidgets.QVBoxLayout(self)
        # layout.addWidget(self.lineedit)
        # layout.setContentsMargins(0, 0, 0, 0)
        # self.setLayout(layout)

    def setPopupWidth(self, width):
        """Set the width of the popup to match the line edit"""
        self.lineedit.completer().popup().setMinimumWidth(width + 50)
        
        self.lineedit.completer().popup().setMaximumHeight(10)
        self.lineedit.completer().popup().setMinimumHeight(10)


class AnotherWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(AnotherWidget, self).__init__(parent)
        
        self.autocomplete_widget = AutocompleteWidget(self)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.autocomplete_widget)
        
        other_label = QtWidgets.QLabel("This is another widget", self)
        layout.addWidget(other_label)
        
        self.setLayout(layout)
        
        # Set the popup width to match the line edit
        self.autocomplete_widget.setPopupWidth(self.autocomplete_widget.lineedit.width() + 50)
        
    # def handle_search(self, data, text):
    #     # Implement your search handling here
    #     pass


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = AnotherWidget()
    w.resize(400, 200)
    w.show()
    sys.exit(app.exec_())