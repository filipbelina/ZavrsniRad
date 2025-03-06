import random
import sys
from ai_agent import AIAgent
import pygame

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
        ['..O..',
         '..O..',
         '..O..',
         '..O..',
         '.....'],
        ['.OOOO',
         '.....',
         '.....',
         '.....',
         '.....'],
        ['..O..',
         '..O..',
         '..O..',
         '..O..',
         '.....'],
        ['.OOOO',
         '.....',
         '.....',
         '.....',
         '.....'],
    ],
    [
        ['.....',
         '.....',
         '..O..',
         '.OOO.',
         '.....'],
        ['.....',
         '..O..',
         '.OO..',
         '..O..',
         '.....'],
        ['.....',
         '.....',
         '.OOO.',
         '..O..',
         '.....'],
        ['.....',
         '..O..',
         '..OO.',
         '..O..',
         '.....']
    ],
    [
        [
            '..OO.',
            '.OO..',
            '.....',
            '.....',
            '.....'],
        [
         '..O..',
         '..OO.',
         '...O.',
         '.....',
        '.....'],
        [
            '..OO.',
            '.OO..',
            '.....',
            '.....',
            '.....'],
        [
            '..O..',
            '..OO.',
            '...O.',
            '.....',
            '.....']
    ],
    [
        [
            '.....',
            '.OO..',
            '..OO.',
            '.....',
            '.....'],
        ['.....',
         '...O.',
         '..OO.',
         '..O..',
         '.....'],
        [
            '.....',
            '.OO..',
            '..OO.',
            '.....',
            '.....'],
        ['.....',
         '...O.',
         '..OO.',
         '..O..',
         '.....']
    ],
    [
        ['.....',
         '..O..',
         '..O.',
         '..OO.',
         '.....'],
        ['.....',
         '...O.',
         '.OOO.',
         '.....',
         '.....'],
        ['.....',
         '.OO..',
         '..O..',
         '..O..',
         '.....'],
        ['.....',
         '.....',
         '.OOO.',
         '.O...',
         '.....']
    ],
[
        ['.....',
         '..O..',
         '..O.',
         '.OO..',
         '.....'],
        ['.....',
         '.OOO.',
         '...O.',
         '.....',
         '.....'],
        ['.....',
         '.OO..',
         '.O...',
         '.O...',
         '.....'],
        ['.....',
         '.....',
         '.O...',
         '.OOO.',
         '.....']
    ],
    [
        ['.....',
         '.....',
         '..OO.',
         '..OO.',
         '.....'],
        ['.....',
         '.....',
         '..OO.',
         '..OO.',
         '.....'],
        ['.....',
         '.....',
         '..OO.',
         '..OO.',
         '.....'],
        ['.....',
         '.....',
         '..OO.',
         '..OO.',
         '.....']
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


class Tetris:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
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

    def get_total_height(self):
        highest_point = 0
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    highest_point = max(highest_point, self.height - row)
                    break
        return highest_point

    def get_average_height(self):
        total_height = 0
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    total_height += self.height - row
                    break
        return total_height / self.width if self.width > 0 else 0

    def evaluate_fitness(self):
        return 10000*self.fourCleared+5000*self.threeCleared+500*self.twoCleared+100*self.oneCleared-10*self.total_height-200*self.holes-1*self.average_height

    def count_holes(self):
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    found_block = True
                elif found_block and self.grid[row][col] == 0:
                    holes += 1
        return holes

    def update_trackers(self):
        self.previous_total_height = self.total_height
        self.previous_average_height = self.average_height
        self.previous_holes = self.holes

        self.total_height = self.get_total_height()
        self.average_height = self.get_average_height()
        self.holes = self.count_holes()
        self.fitness = self.evaluate_fitness()

        if self.trackers_changed():
            print(f"Total Height: {self.total_height}")
            print(f"Average Height: {self.average_height:.2f}")
            print(f"Holes: {self.holes}")
            print(f"Fitness: {self.fitness:.2f}")
            print(f"Grid: {self.grid}")
            print("------------------------------")

    def trackers_changed(self):
        return (self.total_height != self.previous_total_height or
                self.average_height != self.previous_average_height or
                self.holes != self.previous_holes)

    def generate_tetromino_bag(self):
        """Generate a list of 14 Tetrominos, 2 of each shape"""
        bag = []
        for shape in SHAPES:
            bag.append(Tetromino(0, 0, shape))
            bag.append(Tetromino(0, 0, shape))
        random.shuffle(bag)
        return bag

    def new_piece(self):
        """Choose a new Tetromino from the bag and regenerate the bag if empty"""
        if not self.tetromino_bag:
            self.tetromino_bag = self.generate_tetromino_bag()
        piece = self.tetromino_bag.pop()
        shape_width = max(len(row) for row in piece.shape[0])
        piece.x = (self.width - shape_width) // 2
        piece.y = 0
        return piece

    def valid_move(self, piece, x, y, rotation):
        """Check if the piece can move to the given position"""
        for i, row in enumerate(piece.shape[(piece.rotation + rotation) % len(piece.shape)]):
            for j, cell in enumerate(row):
                if cell == 'O':
                    new_x = piece.x + j + x
                    new_y = piece.y + i + y
                    # Check horizontal boundaries
                    if new_x < 0 or new_x >= self.width:
                        return False
                    # Check vertical boundaries
                    if new_y < 0 or new_y >= self.height:
                        return False
                    # Check if the cell is occupied
                    if self.grid[new_y][new_x] != 0:
                        return False
        return True

    def clear_lines(self):
        lines_cleared = 0
        for i, row in enumerate(self.grid):
            if all(cell != 0 for cell in row):
                lines_cleared += 1
                del self.grid[i]
                self.grid.insert(0, [0 for _ in range(self.width)])
        return lines_cleared

    def lock_piece(self, piece):
        """Lock the piece in place and create a new piece"""
        for i, row in enumerate(piece.shape[piece.rotation % len(piece.shape)]):
            for j, cell in enumerate(row):
                if cell == 'O':
                    self.grid[piece.y + i][piece.x + j] = piece.color

        # Clear the lines and update the score
        lines_cleared = self.clear_lines()
        if lines_cleared == 1:
            self.score += 100
            self.oneCleared+=1
        elif lines_cleared == 2:
            self.score += 250
            self.twoCleared+=1
        elif lines_cleared == 3:
            self.score += 500
            self.threeCleared+=1
        elif lines_cleared == 4:
            self.score += 1000
            self.fourCleared+=1
        # Create a new piece
        self.current_piece = self.new_piece()
        # Check if the game is over
        if not self.valid_move(self.current_piece, 0, 0, 0):
            self.game_over = True
        return lines_cleared

    def update(self):
        """Move the tetromino down one cell"""
        if not self.game_over:
            if self.valid_move(self.current_piece, 0, 1, 0):
                self.current_piece.y += 1
            else:
                self.lock_piece(self.current_piece)

    def draw(self, screen):
        """Draw the grid and the current piece"""
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, cell, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))

        if self.current_piece:
            for i, row in enumerate(
                    self.current_piece.shape[self.current_piece.rotation % len(self.current_piece.shape)]):
                for j, cell in enumerate(row):
                    if cell == 'O':
                        pygame.draw.rect(screen, self.current_piece.color, (
                            (self.current_piece.x + j) * GRID_SIZE, (self.current_piece.y + i) * GRID_SIZE,
                            GRID_SIZE - 1,
                            GRID_SIZE - 1))