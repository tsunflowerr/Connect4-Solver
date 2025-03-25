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
                print("", gameBoard[i][j], end = '|')
            elif(gameBoard[i][j] == "🔴"):
                print("", gameBoard[i][j], end = '|')
            else:
                print(" ", gameBoard[i][j], end = ' |')
    print("\n   +---+---+---+---+---+---+---+")

def modifyArray(spacePicked, turn) :
    gameBoard[spacePicked[0]][spacePicked[1]] = turn
    
def checkForWinner(chip):
    def check_line(x1, y1, x2, y2, x3, y3, x4, y4):
        return (gameBoard[y1][x1] == chip and 
                gameBoard[y2][x2] == chip and 
                gameBoard[y3][x3] == chip and 
                gameBoard[y4][x4] == chip)

    # Check horizontal lines
    for y in range(rows):
        for x in range(cols - 3):
            if check_line(x, y, x+1, y, x+2, y, x+3, y):
                print("\n Game over!", chip, "wins!")
                return True

    # Check vertical lines
    for x in range(cols):
        for y in range(rows - 3):
            if check_line(x, y, x, y+1, x, y+2, x, y+3):
                print("\n Game over!", chip, "wins!")
                return True

    # Check diagonal (top-left to bottom-right)
    for x in range(cols - 3):
        for y in range(rows - 3):
            if check_line(x, y, x+1, y+1, x+2, y+2, x+3, y+3):
                print("\n Game over!", chip, "wins!")
                return True

    # Check diagonal (bottom-left to top-right)
    for x in range(cols - 3):
        for y in range(3, rows):
            if check_line(x, y, x+1, y-1, x+2, y-2, x+3, y-3):
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
    
def gravityChecker(intendedCoordinate):
    #calculate space below 
    spaceBelow = [None] * 2
    spaceBelow[0] = intendedCoordinate[0] + 1
    spaceBelow[1] = intendedCoordinate[1]
    # Is the coordinate at ground level?
    if(spaceBelow[0] > 5):
        return True
    #Check if there is a chip below
    if (isSpaceAvailabel(spaceBelow) == False):
        return True
    return False
leaveLoop = False 
turnCounter = 0
while leaveLoop == False:
    if(turnCounter % 2 == 0):
        printGameBoard()
        while True:
            spacePicked = input(" \n Player 1, pick a space: ")
            coordinate = coordinateParser(spacePicked)
            try:
                ## Check if the space is available
                if(isSpaceAvailabel(coordinate) == True and gravityChecker(coordinate) == True):
                    modifyArray(coordinate, "🔵")
                    break
                else:
                    print("Not a valid coordinate")
            except:
                print("error occured")
        winner = checkForWinner("🔵")
        turnCounter += 1
    else:
        while True:
            spacePicked = random.choice(possibleLetters) + str(random.randint(0, 5))
            coordinate = coordinateParser(spacePicked)
            try:
                ## Check if the space is available
                if(isSpaceAvailabel(coordinate) == True and gravityChecker(coordinate) == True):
                    modifyArray(coordinate, "🔴")
                    break
            except:
                print("error occured")
        winner = checkForWinner("🔴")
        turnCounter += 1
    if(winner == True):
        printGameBoard()
        break