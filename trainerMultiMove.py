import copy
import pickle
import random
import numpy as np

from game import Tetris


def train_multi_move(trainer, start_move=0):
    moves_made = start_move
    trainer.current_game_state = None
    overall_best_fitness = float('-inf')
    best_nn = None
    no_improvement_generations = 0

    while moves_made < trainer.max_moves:
        for i in range(10):
            fitness_scores_and_games = [multi_move_evaluate_fitness_nn(trainer, nn, trainer.current_game_state) for nn in trainer.population]
            fitness_scores = [score for score, _ in fitness_scores_and_games]
            games = [game for _, game in fitness_scores_and_games]
            sorted_population = [nn for _, nn in
                                 sorted(zip(fitness_scores, trainer.population), key=lambda x: x[0], reverse=True)]
            # Elitism: Keep the top 10% of the population
            elite_count = max(1, trainer.population_size // 10)
            new_population = sorted_population[:elite_count]
            # Crossover: Create children from the top 50% of the population
            for _ in range(trainer.population_size - elite_count):
                parent1, parent2 = random.sample(sorted_population[:trainer.population_size // 2], 2)
                child = trainer.crossover(parent1, parent2)
                child.mutate(trainer.mutation_rate)
                new_population.append(child)
            trainer.population = new_population
            current_best_fitness = max(fitness_scores)
            best_game = games[fitness_scores.index(current_best_fitness)]

            # Update the overall best fitness and NN
            if current_best_fitness > overall_best_fitness:
                overall_best_fitness = current_best_fitness
                best_nn = sorted_population[0]  # Best NN corresponds to the highest fitness
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            if no_improvement_generations >= 500:
                print("No improvement for 15 generations. Ending training.")
                moves_made= trainer.max_moves
                break

        best_game.render()
        print(best_game.column_height)

        trainer.current_game_state = best_game
        print(f'Best Fitness after {moves_made} moves: {current_best_fitness}')
        moves_made += 3

    # Save the best NN after training
    with open('best_nn.pkl', 'wb') as f:
        pickle.dump(best_nn, f)
    print('Training complete, best model saved.')

def multi_move_evaluate_fitness_nn(trainer, nn, game_state=None):
    if game_state is None:
        game = Tetris(10, 20)
    else:
        game = copy.deepcopy(game_state)
    moves=0
    for _ in range(3):
        if game.game_over:

            break
        legal_moves = trainer.get_legal_moves(game)
        if not legal_moves:
            break

        if trainer.training_algorithm == 1 or trainer.training_algorithm == 2:
            inputs = trainer.prepare_inputs1(game)
        else:
            inputs = trainer.prepare_inputs2(game)

        outputs = nn.forward(inputs)
        sorted_indices = np.argsort(outputs)[::-1]
        move_index = None
        for idx in sorted_indices:
            if idx < len(legal_moves):
                move_index = idx
                break
        if move_index is None:
            move_index = sorted_indices[0]
        move = legal_moves[move_index]
        game.current_piece.x += move[0]
        game.current_piece.rotation = (game.current_piece.rotation + move[1]) % len(game.current_piece.shape)
        while game.valid_move(game.current_piece, 0, 1, 0):
            game.current_piece.y += 1
        game.lock_piece(game.current_piece)  # odradi lock peace metodu.
        game.update_trackers()
        moves += 1

    if trainer.fitness_function == 1:
        return game.evaluate_fitness1(moves), game
    elif trainer.fitness_function == 2:
        return game.evaluate_fitness2(moves), game
    elif trainer.fitness_function == 3:
        return game.evaluate_fitness3(moves), game
    else:
        raise ValueError("Invalid training algorithm specified.")