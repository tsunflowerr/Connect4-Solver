import random

print("Welcome to Connect 4!")
print("--------------------")

possibleLetters = ["A", "B", "C", "D", "E", "F", "G"]
gameBoard = [["", "", "", "", "", "", ""],
             ["", "", "", "", "", "", ""],
             ["", "", "", "", "", "", ""],
             ["", "", "", "", "", "", ""],
             ["", "", "", "", "", "", ""],
             ["", "", "", "", "", "", ""]]
rows = 6
cols = 7

def printGameBoard(): 
    print("\n     A   B   C   D   E   F   G", end = "")
    for i in range(rows):
        print("\n   +---+---+---+---+---+---+---+")
        print(i, ' |', end = '')
        for j in range(cols):
            if(gameBoard[i][j] == "🔵"):
                print("", gameBoard[i][j], end = ' |')
            elif(gameBoard[i][j] == "🔴"):
                print("", gameBoard[i][j], end = ' |')
            else:
                print(" ", gameBoard[i][j], end = ' |')
    print("\n   +---+---+---+---+---+---+---+")

def modifyTurn(spacePicked, turn) :
    gameBoard[spacePicked[0]][spacePicked[1]] = turn

printGameBoard()