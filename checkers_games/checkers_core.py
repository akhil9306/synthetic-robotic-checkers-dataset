# importing the libraries
import numpy as np
import random
import copy
import torch
import torch.nn as nn

BOARD_SIZE = 8

# ========================
# Model Definitions
# ========================

# Model 1: Linear model (same as the original notebook — single linear layer)
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 1)

    def forward(self, x):
        return self.linear(x)


# Model 2: Deeper neural network (2 hidden layers)
class MediumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# Model 3: Even deeper neural network (4 hidden layers)
class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


# ========================
# Board & Game Logic
# ========================

# initialise the board
def initializeBoard():
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    pinRows = (BOARD_SIZE // 2) - 1
    for i in range(0, pinRows):
        for j in range(0, BOARD_SIZE):
            if (i + j) % 2 == 1:
                board[i][j] = 'B'
    for i in range(BOARD_SIZE - pinRows, BOARD_SIZE):
        for j in range(0, BOARD_SIZE):
            if (i + j) % 2 == 1:
                board[i][j] = 'R'
    return board


# check if the game is over (returns True if pin has won, i.e., no opponent pieces remain)
def isGameOver(board, pin):
    kingPin = 'K' + pin
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != ' ' and board[i][j] != pin and board[i][j] != kingPin:
                return False
    return True


# check if we can jump from the position x,y to x2,y2 by taking down the opponent
def canJump(board, pin, x, y, x1, y1, x2, y2):
    if x2 < 0 or x2 >= BOARD_SIZE or y2 < 0 or y2 >= BOARD_SIZE:
        return False
    if board[x2][y2] != ' ':
        return False
    if pin == 'R' or pin == 'KR':
        # Middle cell must have an opponent piece (B or KB)
        if board[x1][y1] not in ('B', 'KB'):
            return False
        # Regular R piece cannot jump backward (downward, increasing row)
        if pin == 'R' and x2 > x:
            return False
        return True
    if pin == 'B' or pin == 'KB':
        # Middle cell must have an opponent piece (R or KR)
        if board[x1][y1] not in ('R', 'KR'):
            return False
        # Regular B piece cannot jump backward (upward, decreasing row)
        if pin == 'B' and x2 < x:
            return False
        return True
    return False


# check if we can move from x,y position to x1,y1 position
def canMove(board, pin, x, y, x1, y1):
    if x1 < 0 or x1 >= BOARD_SIZE or y1 < 0 or y1 >= BOARD_SIZE:
        return False
    if board[x1][y1] != ' ':
        return False
    if pin == 'R' and x1 > x:
        return False
    if pin == 'B' and x1 < x:
        return False
    return True


# make all the pins kings if they reach the opposite end for the first time
def makeKing(board):
    for i in range(BOARD_SIZE):
        if board[0][i] == 'R':
            board[0][i] = 'KR'
        if board[BOARD_SIZE - 1][i] == 'B':
            board[BOARD_SIZE - 1][i] = 'KB'
    return board


# check for the possible moves in all the directions
# first we check for jumping and then we check for moving
def possibleMoves(board, pin):
    legalMoves = []
    directions = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
    kingPin = 'K' + pin
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == pin or board[i][j] == kingPin:
                for [ud, lr] in directions:
                    if canJump(board, board[i][j], i, j, i + ud, j + lr, i + 2 * ud, j + 2 * lr):
                        tempBoard = copy.deepcopy(board)
                        tempBoard[i][j] = ' '
                        tempBoard[i + ud][j + lr] = ' '
                        tempBoard[i + 2 * ud][j + 2 * lr] = board[i][j]
                        makeKing(tempBoard)
                        legalMoves.append(tempBoard)
    if len(legalMoves) == 0:
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == pin or board[i][j] == kingPin:
                    for [ud, lr] in directions:
                        if canMove(board, board[i][j], i, j, i + ud, j + lr):
                            tempBoard = copy.deepcopy(board)
                            tempBoard[i][j] = ' '
                            tempBoard[i + ud][j + lr] = board[i][j]
                            makeKing(tempBoard)
                            legalMoves.append(tempBoard)
    return legalMoves


# ========================
# Feature Extraction
# ========================

# here we get the features for the next moves
# here we upweigh the features for pin1 and downweigh the features for pin2
def getFeatures(board, pin1, pin2):
    kingPin1 = 'K' + pin1
    kingPin2 = 'K' + pin2
    countPins = np.zeros(7)
    countPins[0] = 1  # bias term
    directions = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == pin1:
                countPins[1] += 1
            elif board[i][j] == pin2:
                countPins[2] -= 1
            elif board[i][j] == kingPin1:
                countPins[3] += 1
            elif board[i][j] == kingPin2:
                countPins[4] -= 1
            if board[i][j] == pin1 or board[i][j] == kingPin1:
                for [ud, lr] in directions:
                    if canJump(board, board[i][j], i, j, i + ud, j + lr, i + 2 * ud, j + 2 * lr):
                        countPins[5] += 1
            elif board[i][j] == pin2 or board[i][j] == kingPin2:
                for [ud, lr] in directions:
                    if canJump(board, board[i][j], i, j, i + ud, j + lr, i + 2 * ud, j + 2 * lr):
                        countPins[6] -= 1
    return countPins


# ========================
# Display
# ========================

# print the board with row and column numbers
def printBoard(board):
    print("   ", end="")
    for j in range(BOARD_SIZE):
        print(f" {j}  ", end="")
    print()
    print("   " + "----" * BOARD_SIZE)
    for i in range(BOARD_SIZE):
        print(f"{i} |", end="")
        for j in range(BOARD_SIZE):
            piece = board[i][j]
            if piece == ' ':
                print("   |", end="")
            else:
                print(f" {piece:2s}|", end="")
        print()
        print("   " + "----" * BOARD_SIZE)
    print()


# ========================
# Scoring & Move Selection
# ========================

# determining the final score of the game
def getFinalBoardScore(board, pin1, pin2):
    if isGameOver(board, pin1):
        return 100.0
    elif isGameOver(board, pin2):
        return -100.0
    return 0.0


# get board score using the neural network model
def getBoardScore(model, board, pin1, pin2, device):
    features = getFeatures(board, pin1, pin2)
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(features_tensor)
    return score.item()


# here we choose the best possible move from our possible moves
# scored by evaluating each candidate board state using the model
def chooseMove(model, board, pin1, pin2, device):
    legalMoves = possibleMoves(board, pin1)
    if len(legalMoves) == 0:
        return None
    scores = [getBoardScore(model, move, pin1, pin2, device) for move in legalMoves]
    return legalMoves[np.argmax(scores)]


# here we choose a random move (used only for training purposes)
def chooseRandomMove(board, pin):
    legalMoves = possibleMoves(board, pin)
    if len(legalMoves) == 0:
        return None
    return random.choice(legalMoves)


# choose move with epsilon-greedy exploration for training
def chooseMoveEpsilonGreedy(model, board, pin1, pin2, device, epsilon=0.1):
    if random.random() < epsilon:
        return chooseRandomMove(board, pin1)
    return chooseMove(model, board, pin1, pin2, device)


# ========================
# Game Simulation & Training Data
# ========================

# we store the game history here
def getGameHistory(model, board, pin1, pin2, device, epsilon=0.1):
    gameHistory = []
    tempBoard = copy.deepcopy(board)
    maxMoves = 200  # prevent infinite games
    moveCount = 0
    while moveCount < maxMoves:
        # Player 1's turn (uses model with epsilon-greedy)
        tempBoard = chooseMoveEpsilonGreedy(model, tempBoard, pin1, pin2, device, epsilon)
        if tempBoard is None:
            break
        gameHistory.append(copy.deepcopy(tempBoard))
        if isGameOver(tempBoard, pin1):
            break

        # Player 2's turn (random)
        tempBoard = chooseRandomMove(tempBoard, pin2)
        if tempBoard is None:
            break
        gameHistory.append(copy.deepcopy(tempBoard))
        if isGameOver(tempBoard, pin2):
            break

        moveCount += 1
    return gameHistory


# generate training samples from game history using TD learning
def getTrainingSamples(model, pin1, pin2, gameHistory, device):
    trainingSamples = []
    # for each state, the target is the model's evaluation of the successor state
    for i in range(len(gameHistory) - 1):
        features = getFeatures(gameHistory[i], pin1, pin2)
        nextFeatures = getFeatures(gameHistory[i + 1], pin1, pin2)
        nextTensor = torch.FloatTensor(nextFeatures).unsqueeze(0).to(device)
        with torch.no_grad():
            target = model(nextTensor).item()
        trainingSamples.append((features, target))
    # appending the final state — who won the game
    finalFeatures = getFeatures(gameHistory[-1], pin1, pin2)
    finalTarget = getFinalBoardScore(gameHistory[-1], pin1, pin2)
    trainingSamples.append((finalFeatures, float(finalTarget)))
    return trainingSamples


# here we get the game status for each game on who won and who lost or is it a draw
def getGameStatusCount(board, pin1, pin2, gameStatusCount):
    finalScore = getFinalBoardScore(board, pin1, pin2)
    if finalScore == 100.0:
        gameStatusCount[0] += 1
    elif finalScore == -100.0:
        gameStatusCount[1] += 1
    else:
        gameStatusCount[2] += 1
    return gameStatusCount


# ========================
# Human Player Interface
# ========================

# let the human player select a move from the list of legal moves
def humanSelectMove(boardState, humanPin):
    legalMoves = possibleMoves(boardState, humanPin)
    if len(legalMoves) == 0:
        print("No legal moves available!")
        return None

    # display legal moves with indices
    print(f"You have {len(legalMoves)} legal move(s) available:")
    for idx, move in enumerate(legalMoves):
        print(f"\nMove {idx}:")
        printBoard(move)

    # get player's choice
    print(f"\nEnter move number (0-{len(legalMoves) - 1}):")
    moveChoice = -1
    while moveChoice < 0 or moveChoice >= len(legalMoves):
        try:
            moveChoice = int(input())
            if moveChoice < 0 or moveChoice >= len(legalMoves):
                print(f"Invalid choice. Enter a number between 0 and {len(legalMoves) - 1}:")
        except ValueError:
            print(f"Invalid input. Enter a number between 0 and {len(legalMoves) - 1}:")

    return copy.deepcopy(legalMoves[moveChoice])
