import copy
import pickle
import random
import numpy as np

from game import Tetris


def train_multi_move(trainer):
    trainer.current_game_state = None

    generation_stats = []
    total_evaluations = 0

    overall_best_fitness = float('-inf')
    overall_best_nn = None

    for generation in range(1000):
        if total_evaluations >= trainer.evaluations:
            print(f"Reached {trainer.evaluations} evaluations. Stopping training.")
            break
        print(f"Generation {generation}")

        fitness_games = [
            multi_move_evaluate_fitness_nn(trainer, nn, trainer.current_game_state)
            for nn in trainer.population
        ]
        total_evaluations += len(fitness_games)

        fitness_scores = [score for score, _ in fitness_games]
        games = [game for _, game in fitness_games]

        best_idx = int(np.argmax(fitness_scores))
        best_fitness = fitness_scores[best_idx]
        best_game = games[best_idx]
        best_score = best_game.score

        best_game.render()
        best_game.print_stats()
        print(best_game.column_height)

        generation_stats.append(
            f"Generation: {generation + 1}, Fitness: {best_fitness}, Score: {best_score}"
        )

        if best_fitness > overall_best_fitness:
            overall_best_fitness = best_fitness
            overall_best_nn = trainer.population[best_idx]

        if best_score > 150000:
            print(f"Best neural network reached a score of {best_score}. Stopping training.")
            break

        sorted_population = [
            nn for _, nn in
            sorted(zip(fitness_scores, trainer.population), key=lambda x: x[0], reverse=True)
        ]

        elite_count = max(1, trainer.population_size // 10)
        new_population = sorted_population[:elite_count]

        while len(new_population) < trainer.population_size:
            parent1, parent2 = random.sample(sorted_population[:trainer.population_size // 2], 2)
            child = trainer.crossover(parent1, parent2)
            child.mutate(trainer.mutation_rate)
            new_population.append(child)

        trainer.population = new_population

        trainer.current_game_state = best_game

        if (generation + 1) % 5 == 0:
            trainer.best_neural_network = sorted_population[0]
            with open(f"best_per_gen/best_nn_gen_{generation + 1}.pkl", "wb") as f:
                pickle.dump(trainer.best_neural_network, f)
            print(f"Best model saved for generation {generation + 1}.")

    trainer.best_neural_network = overall_best_nn or trainer.population[0]

    with open("best_nn.pkl", "wb") as f:
        pickle.dump(trainer.best_neural_network, f)

    with open("generation_stats.txt", "w") as f:
        f.write("\n".join(generation_stats))

    print("Training complete, best model saved.")

def _best_move_by_simulation(trainer, game, nn):
    legal_moves = trainer.get_legal_moves(game)
    if not legal_moves:
        return None

    best_mv, best_val = None, -np.inf
    for mv in legal_moves:
        sim = game.clone()
        dx, rot = mv
        sim.current_piece.x += dx
        sim.current_piece.rotation = (sim.current_piece.rotation + rot) % len(sim.current_piece.shape)
        while sim.valid_move(sim.current_piece, 0, 1, 0):
            sim.current_piece.y += 1
        sim.lock_piece(sim.current_piece)
        sim.update_trackers()

        if trainer.training_algorithm in (1, 2):
            inputs = trainer.prepare_inputs1(sim)
        else:
            inputs = trainer.prepare_inputs2(sim)

        value = nn.forward(inputs)[0]
        if value > best_val:
            best_mv, best_val = mv, value

    return best_mv

def multi_move_evaluate_fitness_nn(trainer, nn, game_state=None):
    game = copy.deepcopy(game_state) if game_state is not None else Tetris(10, 20)
    moves_made = 0

    for _ in range(3):
        if game.game_over:
            break

        move = _best_move_by_simulation(trainer, game, nn)
        if move is None:
            break

        dx, rot = move
        game.current_piece.x += dx
        game.current_piece.rotation = (game.current_piece.rotation + rot) % len(game.current_piece.shape)
        while game.valid_move(game.current_piece, 0, 1, 0):
            game.current_piece.y += 1
        game.lock_piece(game.current_piece)
        game.update_trackers()
        moves_made += 3

    if trainer.fitness_function == 1:
        fitness = game.evaluate_fitness1(moves_made)
    elif trainer.fitness_function == 2:
        fitness = game.evaluate_fitness2(moves_made)
    elif trainer.fitness_function == 3:
        fitness = game.evaluate_fitness3(moves_made)
    elif trainer.fitness_function == 4:
        fitness = game.evaluate_fitness4(moves_made)
    else:
        raise ValueError("Invalid fitness function selected.")

    return fitness, game
