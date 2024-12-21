"""
dataMod.py
Author: worifung achu
Date: 12.18.2024
Description: This file contains functions for modifying the data dictionary"""

#updates the name of the automata in the data dictionary
def updateName(data, name, newName):
    for automaton in data["Automata"]:
        if automaton["name"] == name:
            automaton["name"] = newName
            return data
#deletes the row of the automata
def deleteRow(data, name, row):
    for automaton in data["Automata"]:
        if automaton["name"] == name:
            actionList = automaton["actions"]
            actionList.pop(row)
            return data

#inserts a new action above the selected row
def insertBelow(data, name, row, newAction):
    for automaton in data["Automata"]:
        if automaton["name"] == name:
            actionList = automaton["actions"]
            actionList.insert(row+1, newAction)
            return data

def editrow(data, name, row, newAction):
    data = deleteRow(data, name, row) #deleting the row first
    data = insertBelow(data, name, row-1, newAction) #inserting the new row
    return data