import numpy as np

# the "game board", with state
class MineSweeper:
    def __init__(self):
        # params
        self.dim1 = 16
        self.dim2 = 30
        self.total_cells = self.dim1 * self.dim2
        self.n_mines = 99
        self.mines = np.zeros([self.dim1, self.dim2])
        self.neighbors = np.zeros([self.dim1, self.dim2])
        self.state = np.zeros([self.dim1, self.dim2])
        self.state.fill(np.nan)
        self.initialized = False
        self.game_over = False
        self.victory = False

    def initialize(self, coordinates):    #not run until after first selection!
        # set up mines
        # randomly place mines anywhere *except* first selected location AND surrounding cells
        # so that first selection is always a 0
        # weird, yeah, but that's how the original minesweeper worked
        available_cells = range(self.total_cells)
        selected = coordinates[0]*self.dim2 + coordinates[1]

        offLimits = np.array([selected-self.dim2-1, selected-self.dim2, selected-self.dim2+1, selected-1, selected, selected+1, selected+self.dim2-1, selected+self.dim2, selected+self.dim2+1])    #out of bounds is ok

        available_cells = np.setdiff1d(available_cells, offLimits)
        self.n_mines = np.minimum(self.n_mines, len(available_cells))  #in case there are fewer remaining cells than mines to place
        mines_flattened = np.zeros([self.total_cells])
        mines_flattened[np.random.choice(available_cells, self.n_mines, replace=False)] = 1
        self.mines = mines_flattened.reshape([self.dim1, self.dim2])

        # set up neighbors
        for i in range(self.dim1):
            for j in range(self.dim2):

                n_neighbors = 0

                for k in range(-1, 2):

                    if i + k >= 0 and i + k < self.dim1:
                        for l in range(-1, 2):

                            if j + l >= 0 and j + l < self.dim2 and (k != 0 or l != 0):
                                n_neighbors += self.mines[i + k, j + l]
                self.neighbors[i, j] = n_neighbors

        self.initialized = True

    def clear_empty_cell(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        self.state[x, y] = self.neighbors[x, y]
        if self.state[x, y] == 0:

            for i in range(-1, 2):

                if x + i >= 0 and x + i < self.dim1:
                    for j in range(-1, 2):

                        if y + j >= 0 and y + j < self.dim2:
                            if np.isnan(self.state[x + i, y + j]):
                                self.clear_empty_cell((x + i, y + j))

    def select_cell(self, coordinates):
        if self.mines[coordinates[0], coordinates[1]] > 0:  #condition always fails on first selection
            self.game_over = True
            self.victory = False

        else:
            if not self.initialized:    #runs after first selection
                self.initialize(coordinates)
            self.clear_empty_cell(coordinates)

            if np.sum(np.isnan(self.state)) == self.n_mines:
                self.game_over = True
                self.victory = True
