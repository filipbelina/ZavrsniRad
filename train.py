import copy

import numpy as np
import random
import pickle
from game import Tetris

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=34):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.weights1 = np.random.randn(hidden_size1, input_size) * 0.2
        self.weights2 = np.random.randn(hidden_size2, hidden_size1) * 0.2
        self.weights3 = np.random.randn(output_size, hidden_size2) * 0.2
        self.bias1 = np.zeros((hidden_size1, 1))
        self.bias2 = np.zeros((hidden_size2, 1))
        self.bias3 = np.zeros((output_size, 1))

    def forward(self, x):
        x = np.array(x).reshape(-1, 1)
        self.z1 = np.dot(self.weights1, x) + self.bias1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.weights2, self.a1) + self.bias2
        self.a2 = np.tanh(self.z2)
        self.z3 = np.dot(self.weights3, self.a2) + self.bias3
        return self.z3.flatten()

    def mutate(self, rate=0.3):
        self.weights1 += np.random.randn(*self.weights1.shape) * rate
        self.weights2 += np.random.randn(*self.weights2.shape) * rate
        self.weights3 += np.random.randn(*self.weights3.shape) * rate
        self.bias1 += np.random.randn(*self.bias1.shape) * rate
        self.bias2 += np.random.randn(*self.bias2.shape) * rate
        self.bias3 += np.random.randn(*self.bias3.shape) * rate

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class EvolutionaryTrainer:
    def __init__(self, population_size=100, generations=100, mutation_rate=0.3, max_moves=100):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_moves = max_moves
        self.population = [NeuralNetwork(232, 120, 80, 34) for _ in range(population_size)]
        self.current_game_state = None

    def evaluate_fitness_all_games(self, nn, game_states):
        total_fitness = 0
        for game_state in game_states:
            fitness, _ = self.evaluate_fitness_nn(nn, game_state)
            total_fitness += fitness
        return total_fitness / len(game_states)

    def evaluate_fitness_nn(self, nn, game_state=None):
        if game_state is None:
            game = Tetris(10, 20)
        else:
            game = copy.deepcopy(game_state)
        moves=0

        for _ in range(2):  # Play 3 moves
            if game.game_over:
                break

            legal_moves = self.get_legal_moves(game)

            if not legal_moves:
                break

            inputs = self.prepare_inputs(game, legal_moves)

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

            # total_fitness += game.evaluate_fitness(moves)

        return game.evaluate_fitness(moves), game

    def get_legal_moves(self, game):
        piece = game.current_piece
        moves = []
        for rotation in range(len(piece.shape)):
            for x_offset in range(-game.width // 2 - 1, game.width // 2 + 1):
                if game.valid_move(piece, x_offset, 0, rotation):
                    moves.append((x_offset, rotation))
        return moves

    def prepare_inputs(self, game, moves):
        binary_grid = np.array(game.binary_grid)

        top_rows = binary_grid.flatten()

        features = [game.average_height, game.total_height]

        # Add the 5x5 array of the first rotation of the current piece
        first_rotation = np.array(game.current_piece.shape[0]).flatten()
        features.extend([game.total_height,game.average_height,game.holes,game.bumpiness,game.totalRowFullness])

        return np.concatenate([top_rows, first_rotation, features])

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train(self, start_move=0):
        no_improvement_generations = 0
        best_fitness = -float('inf')
        moves_made = start_move

        if start_move > 0:
            # Load the saved game state
            with open('saved_game.pkl', 'rb') as f:
                self.current_game_state = pickle.load(f)
        else:
            self.current_game_state = None

        while moves_made < self.max_moves:
            for i in range(20):
                fitness_scores_and_games = [self.evaluate_fitness_nn(nn, self.current_game_state) for nn in self.population]

                fitness_scores = [score for score, _ in fitness_scores_and_games]

                games = [game for _, game in fitness_scores_and_games]

                sorted_population = [nn for _, nn in
                                     sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)]

                # Elitism: Keep the top 10% of the population
                elite_count = max(1, self.population_size // 10)

                new_population = sorted_population[:elite_count]

                # Crossover: Create children from the top 50% of the population
                for _ in range(self.population_size - elite_count):
                    parent1, parent2 = random.sample(sorted_population[:self.population_size // 2], 2)

                    child = self.crossover(parent1, parent2)

                    child.mutate(self.mutation_rate)

                    new_population.append(child)

                self.population = new_population

                current_best_fitness = max(fitness_scores)

            best_game = games[fitness_scores.index(current_best_fitness)]

            # Save the best game state
            with open('saved_game.pkl', 'wb') as f:
                pickle.dump(best_game, f)

            best_game.render()

            self.current_game_state = best_game

            print(f'Best Fitness after {moves_made + 3} moves: {current_best_fitness}')

            moves_made += 3

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            if no_improvement_generations >= 20:
                print('No improvement for 5 generations, stopping early.')
                break

        best_nn = self.population[0]

        with open('best_nn.pkl', 'wb') as f:
            pickle.dump(best_nn, f)

        print('Training complete, best model saved.')

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size1, parent1.hidden_size2, parent1.output_size)
        alpha = np.random.rand()
        child.weights1 = alpha * parent1.weights1 + (1 - alpha) * parent2.weights1
        child.weights2 = alpha * parent1.weights2 + (1 - alpha) * parent2.weights2
        child.weights3 = alpha * parent1.weights3 + (1 - alpha) * parent2.weights3
        child.bias1 = alpha * parent1.bias1 + (1 - alpha) * parent2.bias1
        child.bias2 = alpha * parent1.bias2 + (1 - alpha) * parent2.bias2
        child.bias3 = alpha * parent1.bias3 + (1 - alpha) * parent2.bias3
        return child


if __name__ == "__main__":
    trainer = EvolutionaryTrainer()
    trainer.train()
