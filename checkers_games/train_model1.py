# Training Model 1: Linear Model
# Same architecture as the original notebook — single linear layer (7 -> 1)
# Uses TD learning with MSE loss and Adam optimizer

import torch
import torch.nn as nn
import os
import sys

from checkers_core import (
    LinearModel, initializeBoard, getGameHistory, getTrainingSamples,
    getFeatures, getGameStatusCount
)

# set device (use CUDA if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# training hyperparameters
NUM_TRAINING_GAMES = 2000
LEARNING_RATE = 0.001
EPSILON_START = 0.3
EPSILON_END = 0.05

# initialize the model
model = LinearModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

pins = ('R', 'B')
gameStatusCount = [0, 0, 0]

# training loop
print(f"Training Model 1 (Linear) for {NUM_TRAINING_GAMES} games...\n")
for game in range(NUM_TRAINING_GAMES):
    # decay epsilon over training (explore less as we learn more)
    epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (game / NUM_TRAINING_GAMES)

    # creating the initial board
    board = initializeBoard()

    # Performance System — play a game and record history
    gameHistory = getGameHistory(model, board, pins[0], pins[1], device, epsilon)

    if len(gameHistory) == 0:
        continue

    # Critic — generate training samples from game history
    trainData = getTrainingSamples(model, pins[0], pins[1], gameHistory, device)

    # Generalizer — update weights using MSE loss (replaces LMS rule)
    total_loss = 0.0
    for features, target in trainData:
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        target_tensor = torch.FloatTensor([target]).unsqueeze(0).to(device)

        prediction = model(features_tensor)
        loss = criterion(prediction, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # track game results
    gameStatusCount = getGameStatusCount(gameHistory[-1], pins[0], pins[1], gameStatusCount)

    # print progress every 200 games
    if (game + 1) % 200 == 0:
        print(f"Game {game + 1}/{NUM_TRAINING_GAMES} | "
              f"P1(R) Wins: {gameStatusCount[0]}, P2(B) Wins: {gameStatusCount[1]}, "
              f"Draws: {gameStatusCount[2]} | Epsilon: {epsilon:.3f} | "
              f"Avg Loss: {total_loss / len(trainData):.4f}")

# save the trained weights
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model1_weights.pt')
torch.save(model.state_dict(), save_path)

# the training results
print(f"\nTraining Complete!")
print(f"Results: P1(R) Wins = {gameStatusCount[0]}, "
      f"P2(B) Wins = {gameStatusCount[1]}, "
      f"Draws = {gameStatusCount[2]}")
print(f"Model 1 weights saved to: {save_path}")
