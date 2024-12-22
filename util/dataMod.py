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
    data = data[:row] + data[row+1:]
    return  data

#inserts a new action above the selected row
def insertRow(data, name, row, newAction):
    data = data[:row] + [newAction] + data[row:]
    return data

def editrow(data, name, row, newAction): 
       
    data[row] = newAction

    return data