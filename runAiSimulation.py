import pygame
import pickle
from game import Tetris
from ai_agent import AIAgent
from renderer import draw_score, draw_game_over

WIDTH, HEIGHT = 400, 800
GRID_SIZE = 40
BLACK = (0, 0, 0)

def run_ai_simulation(trainer_path=None, seed=1):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris AI Simulation')

    game = Tetris(10, 20, seed)

    ai_agent = AIAgent(pickle.load(open(trainer_path, 'rb')))

    while not game.game_over:
        screen.fill(BLACK)

        move = ai_agent.find_best_move(game)
        if move:
            dx, rot = move
            game.current_piece.x += dx
            game.current_piece.rotation = (game.current_piece.rotation + rot) % len(game.current_piece.shape)
            while game.valid_move(game.current_piece, 0, 1, 0):
                game.current_piece.y += 1
            game.lock_piece(game.current_piece)
            game.update_trackers()

        draw_score(screen, game.score, 10, 10)
        game.draw(screen)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    draw_game_over(screen, WIDTH // 2 - 100, HEIGHT // 2 - 30)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

if __name__ == "__main__":
    run_ai_simulation()