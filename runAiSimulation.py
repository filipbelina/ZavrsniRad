import time
import pygame
import sys
import pickle
import logging
from game import Tetris
from ai_agent import AIAgent
from renderer import draw_score, draw_game_over

# Constants
WIDTH, HEIGHT = 400, 800
GRID_SIZE = 40
BLACK = (0, 0, 0)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ai_simulation(nn=None, trainer=None):
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris AI Simulation')

    # Create a clock object
    clock = pygame.time.Clock()

    # Create a Tetris object
    game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)

    # Load the trained neural network
    #with open('best_nn.pkl', 'rb') as f:
    #    nn = pickle.load(f)

    ai_agent = AIAgent(nn)

    while not game.game_over:
        screen.fill(BLACK)

        legal_moves = ai_agent.get_legal_moves(game)
        time.sleep(1)

        if legal_moves:
            if trainer.training_algorithm == 1 or trainer.training_algorithm == 2:
                inputs = trainer.prepare_inputs1(game)
            else:
                # print(game.column_height)
                inputs = trainer.prepare_inputs2(game)

            outputs = nn.forward(inputs)
            move_index = ai_agent.select_move(outputs, legal_moves)
            move = legal_moves[move_index]

            logging.debug(f'Move selected: {move}')

            game.current_piece.x += move[0]
            game.current_piece.rotation = (game.current_piece.rotation + move[1]) % len(game.current_piece.shape)

            while game.valid_move(game.current_piece, 0, 1, 0):
                game.current_piece.y += 1

            game.lock_piece(game.current_piece)

        # Draw everything
        draw_score(screen, game.score, 10, 10)
        game.draw(screen)

        pygame.display.flip()
        #clock.tick(60)

    draw_game_over(screen, WIDTH // 2 - 100, HEIGHT // 2 - 30)
    pygame.display.flip()
    time.sleep(3)

if __name__ == "__main__":
    run_ai_simulation()