import random
import sys

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
                try:
                    if cell == 'O' and (self.grid[piece.y + i + y][piece.x + j + x] != 0):
                        return False
                except IndexError:
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
        elif lines_cleared == 2:
            self.score += 250
        elif lines_cleared == 3:
            self.score += 500
        elif lines_cleared == 4:
            self.score += 1000
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


def draw_score(screen, score, x, y):
    """Draw the score on the screen"""
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (x, y))


def draw_game_over(screen, x, y):
    """Draw the game over text on the screen"""
    font = pygame.font.Font(None, 48)
    text = font.render("Game Over", True, RED)
    screen.blit(text, (x, y))


def main():
    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris')

    # Create a clock object
    clock = pygame.time.Clock()

    # Create a Tetris object
    game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)

    fall_time = 0
    fall_speed = 50  # Milliseconds before the piece moves down
    key_delay = 10  # Initial delay before movement repeats (milliseconds)
    key_repeat_delay = 5  # Delay between repeated movements after initial hold

    key_timer = {
        pygame.K_LEFT: {"pressed": False, "time": 0},
        pygame.K_RIGHT: {"pressed": False, "time": 0},
        pygame.K_DOWN: {"pressed": False, "time": 0},
        pygame.K_UP: {"pressed": False, "time": 0}
    }

    while True:
        screen.fill(BLACK)

        # Get time elapsed since last frame
        delta_time = clock.get_rawtime()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    while game.valid_move(game.current_piece, 0, 1, 0):
                        game.current_piece.y += 1  # Drop piece instantly
                    game.lock_piece(game.current_piece)

                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key in key_timer:
                    key_timer[event.key]["pressed"] = True
                    key_timer[event.key]["time"] = 0  # Reset delay
                    # Instant move on key press
                    if event.key == pygame.K_LEFT and game.valid_move(game.current_piece, -1, 0, 0):
                        game.current_piece.x -= 1
                    if event.key == pygame.K_RIGHT and game.valid_move(game.current_piece, 1, 0, 0):
                        game.current_piece.x += 1
                    if event.key == pygame.K_DOWN and game.valid_move(game.current_piece, 0, 1, 0):
                        game.current_piece.y += 1
                    if event.key == pygame.K_UP and game.valid_move(game.current_piece, 0, 0, 1):
                        game.current_piece.rotation += 1

            if event.type == pygame.KEYUP:
                if event.key in key_timer:
                    key_timer[event.key]["pressed"] = False
                    key_timer[event.key]["time"] = 0  # Reset repeat delay

        # Handle held keys
        keys = pygame.key.get_pressed()
        for key in key_timer:
            if key_timer[key]["pressed"]:
                key_timer[key]["time"] += delta_time
                if key_timer[key]["time"] > key_delay:
                    if key_timer[key]["time"] > key_delay + key_repeat_delay:
                        if key == pygame.K_LEFT and game.valid_move(game.current_piece, -1, 0, 0):
                            game.current_piece.x -= 1
                        if key == pygame.K_RIGHT and game.valid_move(game.current_piece, 1, 0, 0):
                            game.current_piece.x += 1
                        if key == pygame.K_DOWN and game.valid_move(game.current_piece, 0, 1, 0):
                            game.current_piece.y += 1
                        if key == pygame.K_UP and game.valid_move(game.current_piece, 0, 0, 1):
                            game.current_piece.rotation += 1
                        key_timer[key]["time"] = key_delay  # Reset for continuous movement

        # Gravity update (moving piece down)
        fall_time += delta_time
        if fall_time >= fall_speed:
            game.update()
            fall_time = 0

        # Draw everything
        draw_score(screen, game.score, 10, 10)
        game.draw(screen)

        # Handle game over
        if game.game_over:
            draw_game_over(screen, WIDTH // 2 - 100, HEIGHT // 2 - 30)
            if event.type == pygame.KEYDOWN:
                game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
