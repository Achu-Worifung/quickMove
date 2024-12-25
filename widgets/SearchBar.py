import xml.etree.ElementTree as ET
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork


class SuggestionModel(QtGui.QStandardItemModel):
    def __init__(self, parent=None):
        super(SuggestionModel, self).__init__(parent)
        self._manager = QtNetwork.QNetworkAccessManager(self)
        self._reply = None
        # self._suggestions = set()  # To track existing suggestions and avoid duplicates

    @QtCore.pyqtSlot(str)
    def search(self, text):
        if self._reply is not None:
            self._reply.abort()
        if text:
            r = self.create_request(text)
            self._reply = self._manager.get(r)
            self._reply.finished.connect(self.on_finished)

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
            content = reply.readAll().data()
            suggestions = ET.fromstring(content)
            self.clear()  # Clear existing suggestions before adding new ones
            for data in suggestions.iter("suggestion"):
                suggestion = data.attrib["data"]
                # if suggestion not in self._suggestions:  # Avoid duplicates
                #     self._suggestions.add(suggestion)
                #     self.appendRow(QtGui.QStandardItem(suggestion))
        reply.deleteLater()
        self._reply = None


class Completer(QtWidgets.QCompleter):
    def splitPath(self, path):
        self.model().search(path)
        return super(Completer, self).splitPath(path)


class AutocompleteWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(AutocompleteWidget, self).__init__(parent)
        
        # Initialize the model and completer for the autocomplete functionality
        self._model = SuggestionModel(self)
        completer = Completer(self, caseSensitivity=QtCore.Qt.CaseInsensitive)
        completer.setModel(self._model)
        
        # Create the QLineEdit and set the completer
        self.lineedit = QtWidgets.QLineEdit(self)
        self.lineedit.setCompleter(completer)

        # Layout to add the QLineEdit to the widget
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.lineedit)
        self.setLayout(layout)


class AnotherWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(AnotherWidget, self).__init__(parent)
        
        # Create an instance of AutocompleteWidget and add it to AnotherWidget
        autocomplete_widget = AutocompleteWidget(self)
        
        # Layout to add the AutocompleteWidget
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(autocomplete_widget)
        
        # Add other widgets to this widget as needed
        other_label = QtWidgets.QLabel("This is another widget", self)
        layout.addWidget(other_label)
        
        self.setLayout(layout)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    # Create an instance of AnotherWidget and display it
    w = AnotherWidget()
    w.resize(400, 200)
    w.show()

    sys.exit(app.exec_())
