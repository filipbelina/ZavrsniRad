import copy
import pickle
import random
import numpy as np

from game import Tetris
from runAiSimulation import run_ai_simulation


def train_full_run(trainer):

    trainer.current_game_state = None
    generation_stats = []

    for i in range(trainer.generations):
        print("Generation {}".format(i))

        fitness_scores_and_games = [fullRunEvaluateFitness(trainer, nn) for nn in trainer.population]

        fitness_scores = [score for score, _ in fitness_scores_and_games]
        games = [game for _, game in fitness_scores_and_games]

        best_game_index = fitness_scores.index(max(fitness_scores))
        best_game = games[best_game_index]

        best_game.render()
        best_game.print_stats()
        print(best_game)

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

        if i % 2 == 0:
            trainer.best_neural_network = trainer.population[0]
            with open(f'best_per_gen/best_nn_gen_{i}.pkl', 'wb') as f:
                pickle.dump(trainer.best_neural_network, f)
            print(f'Best model saved for generation {i}.')

        best_fitness = fitness_scores[best_game_index]
        best_score = best_game.score
        generation_stats.append(f"Generation: {i}, Fitness: {best_fitness}, Score: {best_score}")

    trainer.best_neural_network = trainer.population[0]

    with open('best_nn.pkl', 'wb') as f:
        pickle.dump(trainer.best_neural_network, f)

    with open('generation_stats.txt', 'w') as f:
        f.write("\n".join(generation_stats))
    print(generation_stats)

    print('Training complete, best model saved.')

    run_ai_simulation(trainer)

def fullRunEvaluateFitness(trainer, nn):
    game = Tetris(10, 20)
    moves=0

    while not game.game_over:
        legal_moves = trainer.get_legal_moves(game)
        if not legal_moves:
            break

        best_mv, best_val = None, -np.inf
        for mv in legal_moves:
            sim = copy.deepcopy(game)
            dx, rot = mv
            sim.current_piece.x += dx
            sim.current_piece.rotation = (sim.current_piece.rotation + rot) % len(sim.current_piece.shape)
            while sim.valid_move(sim.current_piece, 0, 1, 0):
                sim.current_piece.y += 1
            sim.lock_piece(sim.current_piece)
            sim.update_trackers()

            if trainer.training_algorithm in (1, 2):
                inp = trainer.prepare_inputs1(sim)
            else:
                inp = trainer.prepare_inputs2(sim)

            val = nn.forward(inp)[0]
            if val > best_val:
                best_mv, best_val = mv, val

        dx, rot = best_mv
        game.current_piece.x += dx
        game.current_piece.rotation = (game.current_piece.rotation + rot) % len(game.current_piece.shape)
        while game.valid_move(game.current_piece, 0, 1, 0):
            game.current_piece.y += 1
        game.lock_piece(game.current_piece)
        game.update_trackers()

        moves += 1

    #game.render()
    if(trainer.fitness_function == 1):
        return game.evaluate_fitness1(moves), game
    elif(trainer.fitness_function == 2):
        return game.evaluate_fitness2(moves), game
    elif(trainer.fitness_function == 3):
        return game.evaluate_fitness3(moves), game
    else:
        raise ValueError("Invalid fitness function selected.")