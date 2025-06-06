import time

import pygame
import sys
import pickle
from game import Tetris
from ai_agent import AIAgent
from renderer import draw_score, draw_game_over
from neuralNetwork import NeuralNetwork

# Constants
WIDTH, HEIGHT = 400, 800
GRID_SIZE = 40
BLACK = (0, 0, 0)

def main():
    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris')

    # Create a clock object
    clock = pygame.time.Clock()

    # Create a Tetris object
    game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)

    # Load the trained neural network
    with open('175_jedinki_100_generacija.pkl', 'rb') as f:
        nn = pickle.load(f)

    ai_agent = AIAgent(nn)

    fall_time = 0
    fall_speed = 100  # Milliseconds before the piece moves down
    key_delay = 30  # Initial delay before movement repeats (milliseconds)
    key_repeat_delay = 15  # Delay between repeated movements after initial hold

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
                    if event.key == pygame.K_UP:
                        new_rotation = (game.current_piece.rotation + 1) % len(game.current_piece.shape)
                        if game.valid_move(game.current_piece, 0, 0, 1):
                            game.current_piece.rotation = new_rotation

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
                        if key == pygame.K_UP:
                            new_rotation = (game.current_piece.rotation + 1) % len(game.current_piece.shape)
                            if game.valid_move(game.current_piece, 0, 0, 1):
                                game.current_piece.rotation = new_rotation
                        key_timer[key]["time"] = key_delay  # Reset for continuous movement

        # Gravity update (moving piece down)
        fall_time += delta_time
        if fall_time >= fall_speed:
            game.update()
            fall_time = 0

        game.update_trackers()

        # AI decision making
        legal_moves = ai_agent.get_legal_moves(game)
        if legal_moves:
            inputs = ai_agent.prepare_inputs(game, legal_moves)
            outputs = nn.forward(inputs)
            move_index = ai_agent.select_move(outputs, legal_moves)
            time.sleep(1)
            move = legal_moves[move_index]

            game.current_piece.x += move[0]

            game.current_piece.rotation = (game.current_piece.rotation + move[1]) % len(game.current_piece.shape)

            while game.valid_move(game.current_piece, 0, 1, 0):
                game.current_piece.y += 1

            game.lock_piece(game.current_piece)

        # Draw everything
        draw_score(screen, game.score, 10, 10)
        try:
            game.draw(screen)
        except IndexError as e:
            print(f"IndexError: {e} - Possibly caused by invalid rotation or piece data.")

        # Handle game over
        if game.game_over:
            draw_game_over(screen, WIDTH // 2 - 100, HEIGHT // 2 - 30)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()