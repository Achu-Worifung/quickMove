import xml.etree.ElementTree as ET

from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork


class SuggestionModel(QtGui.QStandardItemModel):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(SuggestionModel, self).__init__(parent)
        self._manager = QtNetwork.QNetworkAccessManager(self)
        self._reply = None

    @QtCore.pyqtSlot(str)
    def search(self, text):
        self.clear()
        if self._reply is not None:
            self._reply.abort()
        if text:
            r = self.create_request(text)
            self._reply = self._manager.get(r)
            self._reply.finished.connect(self.on_finished)
        loop = QtCore.QEventLoop()
        self.finished.connect(loop.quit)
        loop.exec_()

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
            for data in suggestions.iter("suggestion"):
                suggestion = data.attrib["data"]
                self.appendRow(QtGui.QStandardItem(suggestion))
            self.error.emit("")
        elif reply.error() != QtNetwork.QNetworkReply.OperationCanceledError:
            self.error.emit(reply.errorString())
        else:
            self.error.emit("")
        self.finished.emit()
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
        lineedit = QtWidgets.QLineEdit()
        lineedit.setCompleter(completer)
        label = QtWidgets.QLabel()
        self._model.error.connect(label.setText)
        lay = QtWidgets.QFormLayout(self)
        lay.addRow("Location: ", lineedit)
        lay.addRow("Error: ", label)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Widget()
    w.resize(400, w.sizeHint().height())
    w.show()
    sys.exit(app.exec_())