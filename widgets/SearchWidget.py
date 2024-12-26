import os
from dotenv import load_dotenv
import util.getReference as getReference
import aiohttp
import asyncio
from PyQt5.uic import loadUi
import httpx
from PyQt5.QtWidgets import (
    QApplication, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QDialog, QLabel, QPushButton,QHBoxLayout, QRadioButton, QPushButton
)
from PyQt5.uic import loadUi
from widgets.SearchBar import AutocompleteWidget
import  util.savedVerses as savedVerses
class SearchWidget(QDialog):
    def __init__(self, data = None):
        super().__init__()
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/search.ui')
        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"

        loadUi(ui_path, self)

        #adding action event to the version
        self.version.addItems(['','KJV', 'NIV', 'ESV'])
        self.version.setCurrentIndex(0)
        self.data = data
        print('data in search widget', data)    
        
        #getting the search bar
        autoComplete_widget = AutocompleteWidget(self)
        self.horizontalLayout.addWidget(autoComplete_widget)

        #adding change event to line edit
        autoComplete_widget.lineedit.textChanged.connect(
        lambda text, d=self.data: self.searchVerse(d, text)
        )

        
        
        # version.currentIndexChanged.connect(self.searchVerse)
        current_text = self.version.currentText()

        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('SEARCH_ENGINE_ID')
    def searchVerse(self, data=None, query=None, enter=False):
        """
        Handles search logic when the user types in the search bar.
        """
        num_space = query.count(' ')

        # Ensure the query is valid for searching
        if num_space >= 3 or enter:
            # Perform the search
            try:
                link = os.path.join(os.path.dirname(__file__), '../ui/result.ui')
                results = self.google_srch(query=query, api_key=self.api_key, engine_id=self.engine_id)
                # print('results', results['items'])
                #result items contains title, snippet
                all_results = results['items']
                all_results = all_results[::-1]
                for result in all_results:
                    #loading the vere panel from result
                    single_result = loadUi(link)
                    title_array = getReference.getReference(result['title'])
                    body = getReference.boldedText(result['snippet'], query)
                    if len(title_array) == 0:
                        continue
                    title = title_array[0]
                    single_result.body.setText(body)
                    single_result.title.setText(title)

                    self.searchPane.addWidget(single_result)
                    # print('title',title)
                    # print()
                    # print('body',body)
                    # single_result.title.setText(getReference.getReference(result['title']))
                    # print('title',result['title'])
                    # print()
                    # print('snippet',result['snippet'])
                    # print()
                    # print('link',result['link'])
                    # print()

            except Exception as e:
                print('Error during search:', e)
        else:
            print("Insufficient input for search.")

    def google_srch(self, query, api_key, engine_id, num_results=10):
        """
        Performs a Google Custom Search API request.
        """
        print(f'{query=}, {api_key=}, {engine_id=}, {num_results=}')
        url = 'https://customsearch.googleapis.com/customsearch/v1'
        params = {
            "key": api_key,
            "cx": engine_id,
            "q": f'{query}  site:biblegateway.com',
            "num": num_results
        }
        
        # Perform the synchronous HTTP request
        response = httpx.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
