import numpy as np
import os
from MineSweeper import MineSweeper

np.set_printoptions(linewidth=1000, precision=3, suppress=True)


def playMinesweeper():
    os.system("clear")
    print("Welcome to Minesweeper")

    toss = input("Enter your name: ")

    # out of bounds messages
    outOfBoundsMessages = ['index out of range', 'Index out of bounds', 'Woww']

    playing = True
    while playing:
        message = "let's play."
        game = MineSweeper()
        while not game.game_over:
            # os.system("clear")
            print(message)
            print("There are 99 total mines.")
            print(game.state)
            message = "You're still alive, somehow."
            coordinates = input("Enter coordinates: ")
            coordinates = [int(x) for x in coordinates.split(',')]
            # if isinstance(coordinates, tuple) and coordinates[0] > 0 and coordinates[0] <= game.dim1 and coordinates[1] >0 and coordinates[1] <= game.dim2:
            if coordinates[0] > 0 and coordinates[0] <= game.dim1 and coordinates[1] > 0 and coordinates[
                1] <= game.dim2:
                coordinates = (coordinates[0] - 1, coordinates[1] - 1)
                game.select_cell(coordinates)
            else:
                message = outOfBoundsMessages[np.random.randint(0, len(outOfBoundsMessages), 1)[0]]
        if game.victory:
            os.system("clear")
            print(game.state)
            print("I guess sometimes it's better to be lucky than good. Congratulations.")
        else:
            print("Game over")

        again = input("Play again? (y/n)")
        if again != "y":
            playing = False
            print("Your choice")


playMinesweeper()

