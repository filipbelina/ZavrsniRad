import random

import pygame
from game import Tetris
from renderer import draw_score, draw_game_over

WIDTH, HEIGHT = 400, 800
GRID_SIZE = 40
BLACK = (0, 0, 0)

def manual_play():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris")
    game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE, random.random())
    clock = pygame.time.Clock()

    fall_time = 0
    fall_speed = 200
    lock_delay = 500
    lock_timer = 0
    key_delay = 300
    key_repeat_delay = 50
    key_timer = {k: 0 for k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_UP)}
    running = True

    while running:
        delta_time = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    while game.valid_move(game.current_piece, 0, 1, 0):
                        game.current_piece.y += 1
                    game.lock_piece(game.current_piece)
                    lock_timer = 0
                if event.key in key_timer:
                    key_timer[event.key] = key_delay
                    if event.key == pygame.K_LEFT and game.valid_move(game.current_piece, -1, 0, 0):
                        game.current_piece.x -= 1
                    elif event.key == pygame.K_RIGHT and game.valid_move(game.current_piece, 1, 0, 0):
                        game.current_piece.x += 1
                    elif event.key == pygame.K_DOWN and game.valid_move(game.current_piece, 0, 1, 0):
                        game.current_piece.y += 1
                    elif event.key == pygame.K_UP and game.valid_move(game.current_piece, 0, 0, 1):
                        game.current_piece.rotation = (game.current_piece.rotation + 1) % len(game.current_piece.shape)
            elif event.type == pygame.KEYUP and event.key in key_timer:
                key_timer[event.key] = 0

        pressed = pygame.key.get_pressed()
        for k, t in key_timer.items():
            if pressed[k]:
                if t > 0:
                    key_timer[k] = max(0, t - delta_time)
                else:
                    if k == pygame.K_LEFT and game.valid_move(game.current_piece, -1, 0, 0):
                        game.current_piece.x -= 1
                    elif k == pygame.K_RIGHT and game.valid_move(game.current_piece, 1, 0, 0):
                        game.current_piece.x += 1
                    elif k == pygame.K_DOWN and game.valid_move(game.current_piece, 0, 1, 0):
                        game.current_piece.y += 1
                    elif k == pygame.K_UP and game.valid_move(game.current_piece, 0, 0, 1):
                        game.current_piece.rotation = (game.current_piece.rotation + 1) % len(game.current_piece.shape)
                    key_timer[k] = key_repeat_delay

        fall_time += delta_time
        if fall_time >= fall_speed:
            if game.valid_move(game.current_piece, 0, 1, 0):
                game.current_piece.y += 1
                lock_timer = 0
            else:
                lock_timer += fall_time
                if lock_timer >= lock_delay:
                    game.lock_piece(game.current_piece)
                    lock_timer = 0
            fall_time = 0

        game.update_trackers()
        screen.fill(BLACK)
        game.draw(screen)
        draw_score(screen, game.score, 10, 10)

        if game.game_over:
            draw_game_over(screen, WIDTH // 2 - 100, HEIGHT // 2 - 30)
            pygame.display.flip()
            pygame.time.wait(1500)
            game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)
            lock_timer = 0
            continue

        pygame.display.flip()

    pygame.quit()
    return
