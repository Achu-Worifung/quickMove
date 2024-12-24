import xml.etree.ElementTree as ET
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork


class SuggestionModel(QtGui.QStandardItemModel):
    def __init__(self, parent=None):
        super(SuggestionModel, self).__init__(parent)
        self._manager = QtNetwork.QNetworkAccessManager(self)
        self._reply = None
        self._suggestions = set()  # To track existing suggestions and avoid duplicates

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
                if suggestion not in self._suggestions:  # Avoid duplicates
                    self._suggestions.add(suggestion)
                    self.appendRow(QtGui.QStandardItem(suggestion))
        reply.deleteLater()
        self._reply = None


class Completer(QtWidgets.QCompleter):
    def splitPath(self, path):
        self.model().search(path)
        return super(Completer, self).splitPath(path)


class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)
        self._model = SuggestionModel(self)
        completer = Completer(self, caseSensitivity=QtCore.Qt.CaseInsensitive)
        completer.setModel(self._model)

        lineedit = QtWidgets.QLineEdit(self)
        lineedit.setCompleter(completer)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(lineedit)
        self.setLayout(layout)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.resize(400, w.sizeHint().height())
    w.show()
    sys.exit(app.exec_())
