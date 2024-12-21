def clearLayout(layout):
    if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()  # Delete any existing widgets (radio buttons, buttons, etc.)
                    elif item.layout():
                        clearLayout(item.layout())  # Recursively clear any layouts