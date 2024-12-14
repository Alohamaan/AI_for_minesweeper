import numpy as np
from final_nn_for_monesweeper.minesweeper.MineSweeper import MineSweeper
import time
import torch
import os

class MineSweeperLearner:

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.BCELoss()
        self.dim1 = 16
        self.dim2 = 30
        self.totalCells = self.dim1*self.dim2

    # ultimately want to put this in the model so each can extract its own shit
    def get_predictors_from_game_state(self, state) -> np.array:
        out = np.zeros((11, self.dim1, self.dim2))
        # channel 0: cell number "holds contains information", i.e. has been revealed
        out[0] = np.where(np.isnan(state), 0, 1)
        # channel 1: cell is on game board (useful for detecting edges when conv does 0 padding)
        out[1] = np.ones((self.dim1, self.dim2))
        # the numeric channels: one layer each for 0 to 8 neighbors; one-hot encoding
        for i in range(0, 9):
            out[i + 2] = np.where(state == i, 1, 0)

        return out

    def learn_mine_sweeper(self, nSamples, nBatches, nEpochsPerBatch, verbose=True):
        for i in range(nBatches):
            cellsRevealed = 0
            gamesPlayed = 0
            gamesWon = 0
            samplesTaken = 0

            # Initialize arrays to store game data
            X = np.zeros((nSamples, 11, self.dim1, self.dim2))  # 11 channels
            X2 = np.zeros((nSamples, 1, self.dim1, self.dim2))
            y = np.zeros((nSamples, 1, self.dim1, self.dim2))



            while samplesTaken < nSamples:
                print(f"Batch {i}, Samples Taken: {samplesTaken}")
                # Initiate game
                game = MineSweeper()
                game.select_cell((int(self.dim1 / 2), int(self.dim2 / 2)))  # Pick middle on first selection

                while not (game.game_over or samplesTaken == nSamples):
                    # Get data input from game state
                    Xnow = self.get_predictors_from_game_state(game.state)
                    X[samplesTaken] = Xnow
                    X2now = np.array([np.where(Xnow[0] == 0, 1, 0)])
                    X2[samplesTaken] = X2now

                    # Convert to tensors
                    Xnow_tensor = torch.from_numpy(Xnow).float().unsqueeze(0)  # Add batch dimension
                    X2now_tensor = torch.from_numpy(X2now).float().unsqueeze(0)  # Add batch dimension

                    # Make probability predictions
                    with torch.no_grad():  # No need to track gradients for inference
                        out = self.model(Xnow_tensor, X2now_tensor)

                    # Choose best remaining cell
                    ordered_probs = np.argsort(out.detach().cpu().numpy()[0] + Xnow[0], axis=None)
                    selected = ordered_probs[0]
                    selected1 = int(selected / self.dim2)
                    selected2 = selected % self.dim2
                    game.select_cell((selected1, selected2))

                    # Find the truth
                    truth = out
                    truth[0, 0, selected1, selected2] = game.mines[selected1, selected2]
                    y[samplesTaken] = truth.detach().cpu().numpy()  # Store the truth in y
                    samplesTaken += 1

                if game.game_over:
                    gamesPlayed += 1
                    cellsRevealed += self.totalCells - np.sum(np.isnan(game.state))
                    if game.victory:
                        gamesWon += 1

            if gamesPlayed > 0:
                meanCellsRevealed = float(cellsRevealed) / gamesPlayed
                propGamesWon = float(gamesWon) / gamesPlayed

            if verbose:
                print(f"Games played, batch {i}: {gamesPlayed}")
                print(f"Mean cells revealed, batch {i}: {meanCellsRevealed}")
                print(f"Proportion of games won, batch {i}: {propGamesWon}")

            # Convert numpy arrays to tensors for training
            X_tensor = torch.from_numpy(X).float()
            X2_tensor = torch.from_numpy(X2).float()
            y_tensor = torch.from_numpy(y).float()

            # Training loop
            for epoch in range(nEpochsPerBatch):
                self.model.train()  # Set the model to training mode
                self.optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                output = self.model(X_tensor, X2_tensor)

                # Compute loss
                loss = self.criterion(output, y_tensor)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                if verbose and (epoch % 10 == 0):
                    print(f"Batch {i}, Epoch {epoch}, Loss: {loss.item()}")

            # Save the model every 100 batches
            if (i + 1) % 100 == 0:
                torch.save(self.model.state_dict(), self.name)
        print(gamesWon)
        return

    def testMe(self, nGames):
        cellsRevealed = 0
        gamesWon = 0

        for i in range(nGames):
            if (i % 10) == 0:
                print("Playing game " + str(i+1) + "...")
            # initiate game
            game = MineSweeper()
            # pick middle on first selection. better than corner.
            game.select_cell((int(self.dim1 / 2), int(self.dim2 / 2)))

            while not game.game_over:
                # get data input from game state
                Xnow = self.get_predictors_from_game_state(game.state)
                X2now = np.array([np.where(Xnow[0] == 0, 1, 0)])

                # make probability predictions
                out = self.model([np.array([Xnow]), np.array([X2now])])
                # choose best remaining cell

                orderedProbs = np.argsort(out[0][0] + Xnow[0], axis=None)  # add Xnow[0] so that already selected cells aren't chosen
                selected = orderedProbs[0]
                selected1 = int(selected / self.dim2)
                selected2 = selected % self.dim2
                game.select_cell((selected1, selected2))

            cellsRevealed += self.totalCells - np.sum(np.isnan(game.state))

            if game.victory:
                gamesWon += 1

        meanCellsRevealed = float(cellsRevealed) / nGames
        propGamesWon = float(gamesWon) / nGames
        print("Proportion of games won, batch " + str(i) + ": " + str(propGamesWon))
        print("Mean cells revealed, batch " + str(i) + ": " + str(meanCellsRevealed))

        return

    def watchMePlay(self):
        play = True

        while play:
            game = MineSweeper()
            os.system("clear")
            print("Beginning play")
            print("Game board:")
            print(game.state)
            # Make first selection in the middle. Better than corner.
            selected1 = int(self.dim1 / 2)
            selected2 = int(self.dim2 / 2)
            game.select_cell((selected1, selected2))
            time.sleep(0.05)
            os.system("clear")

            while not game.game_over:
                print("Last selection: (" + str(selected1 + 1) + "," + str(selected2 + 1) + ")")
                if 'out' in locals():
                    print("Confidence: " + str(np.round(100 * (1 - np.amin(out.detach().cpu().numpy()[0][0]+ Xnow[0])), 2)) + "%")
                print("Game board:")
                print(game.state)

                Xnow = self.get_predictors_from_game_state(game.state)
                X2now = np.array([np.where(Xnow[0] == 0, 1, 0)])

                # Convert to tensors
                Xnow_tensor = torch.from_numpy(Xnow).float().unsqueeze(0)  # Add batch dimension
                X2now_tensor = torch.from_numpy(X2now).float().unsqueeze(0)  # Add batch dimension

                # Make probability predictions
                out = self.model(Xnow_tensor, X2now_tensor)  # Pass tensors as separate arguments
                # Choose best remaining cell
                orderedProbs = np.argsort(out.detach().cpu().numpy()[0][0] + Xnow[0],
                                          axis=None)  # Add Xnow[0] so that already selected cells aren't chosen
                selected = orderedProbs[0]
                selected1 = int(selected / self.dim2)
                selected2 = selected % self.dim2
                game.select_cell((selected1, selected2))
                time.sleep(0.05)
                os.system("clear")

            print("Last selection: (" + str(selected1 + 1) + "," + str(selected2 + 1) + ")")
            print("Confidence: " + str(np.round(100 * (1 - np.amin(out.detach().cpu().numpy()[0][0] + Xnow[0])), 2)) + "%")
            print("Game board:")
            print(game.state)

            if game.victory:
                print("Victory!")
            else:
                print("Game Over")
            get = input("Watch me play again? (y/n): ")
            if get != "y":
                play = False

        return

