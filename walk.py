"""
walk.py
Author: worifung achu
Date: 12.18.2024
This module provides a function to retrieve and write  data from a JSON file in a user-accessible directory."""

import os
import json

# Use a user-accessible directory
dir_path = os.path.expanduser("~/Quick move")  # This resolves to something like C:\Users\<username>\Quick move
file_path = os.path.join(dir_path, "automata.json")  # Construct the full path for the JSON file

def get_data():
    global dir_path, file_path
    try:
        # Ensure the directory exists
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            print("Directory created")

        # Check if the JSON file exists
        if not os.path.isfile(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)  # Initialize with an empty JSON object
                print("File created")

        # Check if the file is empty before loading
        with open(file_path, "r") as f:
            file_content = f.read().strip()
            if not file_content:
                print("File is empty. Returning empty dictionary.")
                return {}
            return json.loads(file_content)

    except OSError as e:
        print(f"An error occurred: {e}")
        return {}  # Return an empty dictionary if an error occurs
    except json.JSONDecodeError:
        print("Failed to decode JSON. Returning empty dictionary.")
        return {}

# Example usage
# data = get_data()
# print("Data:", data)

def write_data(data):
    global file_path
    with open(file_path, "w") as f:
        json.dump(data, f) 

def main():
    data =  [{'action': 'click', 'location': (1163, 892), 'button': 'Button.left'}, {'action': 'click', 'location': (966, 969), 'button': 'Button.left'}, {'action': 'paste', 'button': 'ctrl+v', 'location': 'clipboard'}, {'action': 'click', 'location': (1191, 657), 'button': 'Button.left'}, {'action': 'click', 'location': (1222, 908), 'button': 'Button.left'}, {'action': 'click', 'location': (1208, 644), 'button': 'Button.left'}, {'action': 'paste', 'button': 'ctrl+v', 'location': 'clipboard'}, {'action': 'click', 'location': (1228, 653), 'button': 'Button.left'}]
    write_data(data)
    # data = get_data()
    # print(data)

if __name__ == '__main__':
    main()
