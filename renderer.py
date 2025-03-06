import pygame

WHITE = (255, 255, 255)


def draw_score(screen, score, x, y):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (x, y))


def draw_game_over(screen, x, y):
    font = pygame.font.Font(None, 50)
    text = font.render("GAME OVER", True, WHITE)
    screen.blit(text, (x, y))
