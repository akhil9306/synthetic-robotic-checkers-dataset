# Inference: Model 3 (Deep NN) — Human plays RED, Computer plays BLACK
# Load trained weights and play interactively

import torch
import copy
import os

from checkers_core import (
    DeepModel, initializeBoard, possibleMoves, chooseMove,
    isGameOver, printBoard, humanSelectMove
)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the trained model
model = DeepModel().to(device)
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model3_weights.pt')
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

print("Model 3 (Deep NN: 7->128->64->32->16->1) loaded successfully!")
print(f"Using device: {device}\n")

# playing the game after training
print("Start Playing with Computer? (y/n)")
ans = input()
while ans == "y":
    boardState = initializeBoard()
    print("\nGame begins!!! You play with Red\n")
    printBoard(boardState)

    while True:
        # Human's turn (Red)
        print("Your Turn (Red):\n")
        boardState = humanSelectMove(boardState, 'R')
        if boardState is None:
            print("No legal moves available for you!")
            break
        print("\nYour move:")
        printBoard(boardState)
        if isGameOver(boardState, 'R'):
            break

        # Computer's turn (Black)
        boardState = chooseMove(model, boardState, 'B', 'R', device)
        if boardState is None:
            print("Computer has no legal moves!")
            break
        print("Computer's Turn (Black):\n")
        printBoard(boardState)
        if isGameOver(boardState, 'B'):
            break

    # determine the winner
    if boardState is not None and isGameOver(boardState, 'R'):
        print("You Win!!!")
    elif boardState is not None and isGameOver(boardState, 'B'):
        print("Computer Wins!!!")
    else:
        print("Game is drawn!!!")

    print("\nDo you want to play another game? (y/n)")
    ans = input()
    if ans != 'y':
        break
