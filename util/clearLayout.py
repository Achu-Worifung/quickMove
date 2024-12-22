def clearLayout(layout):
    if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()  # Delete any existing widgets (radio buttons, buttons, etc.)
                    elif item.layout():
                        clearLayout(item.layout())  # Recursively clear any layouts

#delete function
def backToWelcome(self):
       # Lazy import
        from widgets.Welcome import Welcome
        
        # Clear current widgets from scroll area
        while self.scrollAreaWidgetContents.layout().count():
            item = self.scrollAreaWidgetContents.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Find the main window
        main_window = self
        while main_window.parent():
            main_window = main_window.parent()
        
        # Assuming main window has a stacked widget or central layout
        if hasattr(main_window, 'centralWidget'):
            # Remove old central widget if it exists
            old_widget = main_window.centralWidget()
            if old_widget:
                old_widget.deleteLater()
                
            # Create and set new Welcome widget
            welcome_widget = Welcome()
            main_window.setCentralWidget(welcome_widget)
        else:
            # Alternative approach if using a layout
            main_layout = main_window.layout()
            if main_layout:
                while main_layout.count():
                    item = main_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                main_layout.addWidget(Welcome())
