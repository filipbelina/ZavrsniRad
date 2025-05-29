import pickle
import random
import numpy as np

from game import Tetris
from runAiSimulation import run_ai_simulation


def train_full_run(trainer):

    trainer.current_game_state = None

    for i in range(trainer.generations):
        fitness_scores_and_games = [fullRunEvaluateFitness(trainer, nn) for nn in trainer.population]

        fitness_scores = [score for score, _ in fitness_scores_and_games]
        games = [game for _, game in fitness_scores_and_games]

        best_game_index = fitness_scores.index(max(fitness_scores))
        best_game = games[best_game_index]

        best_game.render()
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

    best_nn = trainer.population[0]

    run_ai_simulation(best_nn, trainer)

    with open('best_nn.pkl', 'wb') as f:
        pickle.dump(best_nn, f)

    print('Training complete, best model saved.')

def fullRunEvaluateFitness(trainer, nn):
    game = Tetris(10, 20)
    moves=0

    while(True):
        if game.game_over:
            break

        legal_moves = trainer.get_legal_moves(game)

        if not legal_moves:
            break

        if trainer.training_algorithm == 1 or trainer.training_algorithm == 2:
            inputs = trainer.prepare_inputs1(game)
        else:
            #print(game.column_height)
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

        #game.render()
        game.update_trackers()
        #print(game.column_height)

        moves+= 1

    #game.render()
    if(trainer.fitness_function == 1):
        return game.evaluate_fitness1(moves), game
    elif(trainer.fitness_function == 2):
        return game.evaluate_fitness2(moves), game
    elif(trainer.fitness_function == 3):
        return game.evaluate_fitness3(moves), game
    else:
        raise ValueError("Invalid fitness function selected.")