import random
import sys
from ai_agent import AIAgent
import pygame
import numpy as np

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 800
GRID_SIZE = 40

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

COLORS = [RED, BLUE, GREEN, YELLOW, CYAN, ORANGE, PURPLE]

# Tetromino shapes
SHAPES = [
    [
        # I- shape
        [
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,1,1,1,1],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,1,1,1,1],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
    ],
    [
        # T- shape
        [
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,1,1,1,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,1,1,1,0],
            [0,0,1,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,1,1,0],
            [0,0,1,0,0],
            [0,0,0,0,0]
        ]
    ],
    [
        # Zreverse- shape
        [
            [0,0,1,1,0],
            [0,1,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,1,0,0],
            [0,0,1,1,0],
            [0,0,0,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,1,1,0],
            [0,1,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,1,0,0],
            [0,0,1,1,0],
            [0,0,0,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ]
    ],
    [
        # Z- shape
        [
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,0,1,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,0,0,1,0],
            [0,0,1,1,0],
            [0,0,1,0,0],
            [0,0,0,0,0]],
        [
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,0,1,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]],
        [
            [0,0,0,0,0],
            [0,0,0,1,0],
            [0,0,1,1,0],
            [0,0,1,0,0],
            [0,0,0,0,0]]
    ],
    [
        # L- shape
        [
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,1,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,0,0,1,0],
            [0,1,1,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,1,1,1,0],
            [0,1,0,0,0],
            [0,0,0,0,0]
        ]
    ],
    [
        # Lreverse- shape
        [
            [0,0,0,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,1,1,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,1,1,1,0],
            [0,0,0,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,1,0,0,0],
            [0,1,0,0,0],
            [0,0,0,0,0]
        ],
        [
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,1,0,0,0],
            [0,1,1,1,0],
            [0,0,0,0,0]
        ]
    ],
    [
        # Cube - shape
        [
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,1,0],
            [0,0,1,1,0],
            [0,0,0,0,0]
        ],
        [   [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,1,0],
            [0,0,1,1,0],
            [0,0,0,0,0]
        ],
        [   [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,1,0],
            [0,0,1,1,0],
            [0,0,0,0,0]
        ],
        [   [0,0,0,0,0],
            [0,0,0,0,0],
            [0,0,1,1,0],
            [0,0,1,1,0],
            [0,0,0,0,0]
        ]
    ]
]


class Tetromino:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = self.determineColour()
        self.rotation = 0

    def determineColour(self):
        if self.shape == SHAPES[0]:
            return RED
        elif self.shape == SHAPES[1]:
            return BLUE
        elif self.shape == SHAPES[2]:
            return GREEN
        elif self.shape == SHAPES[3]:
            return YELLOW
        elif self.shape == SHAPES[4]:
            return CYAN
        elif self.shape == SHAPES[5]:
            return ORANGE
        elif self.shape == SHAPES[6]:
            return PURPLE
        else:
            return WHITE

    def get_rotated_shape(self):
        return self.shape[self.rotation % len(self.shape)]


class Tetris:
    def __init__(self, width, height, seed=1):
        random.seed(seed)
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.binary_grid = [[0 for _ in range(width)] for _ in range(height)]
        self.tetromino_bag = self.generate_tetromino_bag()
        self.current_piece = self.new_piece()
        self.game_over = False
        self.score = 0
        self.total_height = 0
        self.average_height = 0
        self.holes = 0
        self.previous_total_height = 0
        self.previous_average_height = 0
        self.previous_holes = 0
        self.fitness = 0
        self.oneCleared = 0
        self.twoCleared = 0
        self.threeCleared = 0
        self.fourCleared = 0
        self.totalRowFullness = 0
        self.cached_total_height = None
        self.cached_average_height = None
        self.cached_holes = None
        self.bumpiness = 0
        self.variance = 0


    def calculate_bumpiness(self):
        heights = [0] * self.width
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    heights[col] = self.height - row
                    break

        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])

        return bumpiness

    def calculate_row_variance(self):
        row_heights = []
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row][col] != 0:
                    row_heights.append(self.height - row)
                    break

        return np.var(row_heights)

    def render(self):
        for row in self.binary_grid:
            print(' '.join(['#' if cell else '.' for cell in row]))
        print('\n' + '-' * 20 + '\n')

    def get_total_height(self):
        if self.cached_total_height is not None:
            return self.cached_total_height
        total_height = 0
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    total_height += self.height - row
                    break
        self.cached_total_height = total_height
        return total_height

    def get_total_row_fullness(self):
        if self.totalRowFullness is not None:
            return self.totalRowFullness
        total_fullness = 0
        for row in range(self.height):
            fullness = 0
            for col in range(self.width):
                if self.grid[row][col] != 0:
                    fullness += 1
            total_fullness += fullness
        self.totalRowFullness = total_fullness
        return total_fullness

    def get_average_height(self):
        if self.cached_average_height is not None:
            return self.cached_average_height
        total_height = 0
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    total_height += self.height - row
                    break
        self.cached_average_height = total_height / self.width if self.width > 0 else 0
        return self.cached_average_height

    #def evaluate_fitness(self, moves):
    #    return (10*self.fourCleared+
    #            5*self.threeCleared+
    #            4*self.twoCleared+
    #            10*self.oneCleared+
    #            10*self.totalRowFullness-
    #            10*self.total_height-
    #            30*self.holes-
    #            1*self.average_height+
    #            1*moves-
    #            1*self.bumpiness-
    #            1*self.variance)

    def evaluate_fitness(self, moves):
        empty_cells_reward = 0
        for row in self.grid:
            empty_cells = row.count(0)
            if empty_cells <= 2:  # Consider rows with 2 or fewer empty cells
                empty_cells_reward += (10 - empty_cells) * 2

        return -0.51*self.total_height + 12.76*(4*self.fourCleared+3*self.threeCleared+2*self.twoCleared+self.oneCleared) - 0.36*self.holes - 0.18*self.bumpiness + 0.5*moves - 0.3*self.variance+empty_cells_reward

    def count_holes(self):
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    found_block = True
                elif found_block and self.grid[row][col] == 0:
                    holes += 1
        self.cached_holes = holes
        return holes

    def update_trackers(self):
        self.previous_total_height = self.total_height
        self.previous_average_height = self.average_height
        self.previous_holes = self.holes

        self.total_height = self.get_total_height()
        self.average_height = self.get_average_height()
        self.holes = self.count_holes()
        self.totalRowFullness = self.get_total_row_fullness()
        self.bumpiness = self.calculate_bumpiness()
        self.variance = self.calculate_row_variance()
        #self.fitness = self.evaluate_fitness()

        # Reset cache
        self.cached_total_height = None
        self.cached_average_height = None
        self.cached_holes = None

    def print_stats(self):
        print(f"Total Height: {self.total_height}")
        print(f"Average Height: {self.average_height:.2f}")
        print(f"Holes: {self.holes}")
        print(f"Fitness: {self.fitness:.2f}")
        print(f"Grid: {self.grid}")
        print(f"buminess: {self.bumpiness}")
        print(f"variance: {self.variance}")
        print("------------------------------")

    def trackers_changed(self):
        return (self.total_height != self.previous_total_height or
                self.average_height != self.previous_average_height or
                self.holes != self.previous_holes)

    def generate_tetromino_bag(self):
        bag = [Tetromino(0, 0, shape) for shape in SHAPES for _ in range(2)]
        random.shuffle(bag)
        return bag


    def new_piece(self):
        if not self.tetromino_bag:
            self.tetromino_bag = self.generate_tetromino_bag()
        piece = self.tetromino_bag.pop()
        piece.x = (self.width - 5) // 2
        piece.y = 0
        return piece

    def valid_move(self, piece, dx, dy, rotation):
        new_x, new_y = piece.x + dx, piece.y + dy
        new_shape = piece.shape[(piece.rotation + rotation) % len(piece.shape)]
        for i in range(5):
            for j in range(5):
                if new_shape[i][j]:
                    if new_x + j < 0 or new_x + j >= self.width:
                        return False
                    if new_y + i < 0 or new_y + i >= self.height:
                        return False
                    if self.grid[new_y + i][new_x + j]:
                        return False
        return True

    def clear_lines(self):
        """Removes full lines and shifts everything down"""
        lines_cleared = 0
        new_grid = [row for row in self.grid if any(cell == 0 for cell in row)]  # Keep non-full rows
        new_binary_grid = [row for row in self.binary_grid if any(cell == 0 for cell in row)]

        # Calculate number of lines cleared
        lines_cleared = self.height - len(new_grid)

        # Add new empty rows at the top to keep grid size consistent
        while len(new_grid) < self.height:
            new_grid.insert(0, [0] * self.width)
            new_binary_grid.insert(0, [0] * self.width)

        self.grid = new_grid
        self.binary_grid = new_binary_grid
        return lines_cleared

    def lock_piece(self, piece):
        """Locks the current piece in place and spawns a new one"""
        for i in range(5):
            for j in range(5):
                if piece.shape[piece.rotation][i][j] == 1:  # Use 1 instead of 'O'
                    grid_x = piece.x + j
                    grid_y = piece.y + i
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        self.grid[grid_y][grid_x] = piece.color  # Assign color instead of 1
                        self.binary_grid[grid_y][grid_x] = 1

        # Clear lines and update score
        lines_cleared = self.clear_lines()
        if lines_cleared == 1:
            self.score += 100
            self.oneCleared += 1
        elif lines_cleared == 2:
            self.score += 250
            self.twoCleared += 1
        elif lines_cleared == 3:
            self.score += 500
            self.threeCleared += 1
        elif lines_cleared == 4:
            self.score += 1000
            self.fourCleared += 1

        # Generate a new piece
        self.current_piece = self.new_piece()

        # Check if the game is over
        if not self.valid_move(self.current_piece, 0, 0, 0):
            self.game_over = True

        return lines_cleared

    def update(self):
        if not self.game_over:
            if self.valid_move(self.current_piece, 0, 1, 0):
                self.current_piece.y += 1
            else:
                self.lock_piece(self.current_piece)

    def draw(self, screen):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, cell, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))
        for i in range(5):
            for j in range(5):
                if self.current_piece.get_rotated_shape()[i][j]:
                    pygame.draw.rect(screen, self.current_piece.color, (
                        (self.current_piece.x + j) * GRID_SIZE,
                        (self.current_piece.y + i) * GRID_SIZE,
                        GRID_SIZE - 1,
                        GRID_SIZE - 1
                    ))