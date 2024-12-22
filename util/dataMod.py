"""
dataMod.py
Author: worifung achu
Date: 12.18.2024
Description: This file contains functions for modifying the data dictionary"""

#updates the name of the automata in the data dictionary
def updateName(data, name, newName):
    return newName
#deletes the row of the automata
def deleteRow(data, name, row):
    return  data.pop(row)

#inserts a new action above the selected row
def insertBelow(data, name, row, newAction):
    return data.insert(row+1, newAction)

def editrow(data, name, row, newAction):    
    data = data[row-1] = newAction

    return data