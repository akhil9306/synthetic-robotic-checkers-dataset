# Inference: Model 1 (Linear) — Human plays BLACK, Computer plays RED
# Load trained weights and play interactively

import torch
import copy
import os

from checkers_core import (
    LinearModel, initializeBoard, possibleMoves, chooseMove,
    isGameOver, printBoard, humanSelectMove
)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the trained model
model = LinearModel().to(device)
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model1_weights.pt')
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

print("Model 1 (Linear: 7->1) loaded successfully!")
print(f"Using device: {device}\n")

# playing the game after training
print("Start Playing with Computer? (y/n)")
ans = input()
while ans == "y":
    boardState = initializeBoard()
    print("\nGame begins!!! You play with Black\n")
    printBoard(boardState)

    while True:
        # Computer's turn (Red — moves first)
        boardState = chooseMove(model, boardState, 'R', 'B', device)
        if boardState is None:
            print("Computer has no legal moves!")
            break
        print("Computer's Turn (Red):\n")
        printBoard(boardState)
        if isGameOver(boardState, 'R'):
            break

        # Human's turn (Black)
        print("Your Turn (Black):\n")
        boardState = humanSelectMove(boardState, 'B')
        if boardState is None:
            print("No legal moves available for you!")
            break
        print("\nYour move:")
        printBoard(boardState)
        if isGameOver(boardState, 'B'):
            break

    # determine the winner
    if boardState is not None and isGameOver(boardState, 'B'):
        print("You Win!!!")
    elif boardState is not None and isGameOver(boardState, 'R'):
        print("Computer Wins!!!")
    else:
        print("Game is drawn!!!")

    print("\nDo you want to play another game? (y/n)")
    ans = input()
    if ans != 'y':
        break
