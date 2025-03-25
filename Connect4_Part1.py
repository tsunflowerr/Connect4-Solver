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
    
def checkForWinner(chip):
    def check_line(x1, y1, x2, y2, x3, y3, x4, y4):
        return (gameBoard[x1][y1] == chip and 
                gameBoard[x2][y2] == chip and 
                gameBoard[x3][y3] == chip and 
                gameBoard[x4][y4] == chip)

    # Kiểm tra hàng ngang
    for y in range(rows):
        for x in range(cols - 3):
            if check_line(x, y, x+1, y, x+2, y, x+3, y):
                print("\n Game over!", chip, "wins!")
                return True

    # Kiểm tra hàng dọc
    for x in range(cols):
        for y in range(rows - 3):
            if check_line(x, y, x, y+1, x, y+2, x, y+3):
                print("\n Game over!", chip, "wins!")
                return True

    # Kiểm tra đường chéo từ phải trên xuống trái dưới
    for x in range(rows - 3):
        for y in range(3, cols):
            if check_line(x, y, x+1, y-1, x+2, y-2, x+3, y-3):
                print("\n Game over!", chip, "wins!")
                return True

    # Kiểm tra đường chéo từ trái trên xuống phải dưới
    for x in range(rows - 3):
        for y in range(cols - 3):
            if check_line(x, y, x+1, y+1, x+2, y+2, x+3, y+3):
                print("\n Game over!", chip, "wins!")
                return True

    return False


def coordinateParser(inputString):
    coordinate = [None] * 2
    if(inputString[0] == "A"):
        coordinate[1] = 0
    elif(inputString[0] == "B"):
        coordinate[1] = 1
    elif(inputString[0] == "C"):
        coordinate[1] = 2
    elif(inputString[0] == "D"):
        coordinate[1] = 3
    elif(inputString[0] == "E"):
        coordinate[1] = 4
    elif(inputString[0] == "F"):
        coordinate[1] = 5
    elif(inputString[0] == "G"):
        coordinate[1] = 6
    else : print("Invalid input")
    coordinate[0] = int(inputString[1])
    return coordinate

def isSpaceAvailabel(intendedCoordinate):
    if(gameBoard[intendedCoordinate[0]][intendedCoordinate[1]] == "🔴"):
        return False
    elif(gameBoard[intendedCoordinate[0]][intendedCoordinate[1]] == "🔵"):
        return False
    else:
        return True
turnCounter = 0

            
    