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
        self.weights1 = np.random.randn(hidden_size1, input_size) * 0.01
        self.weights2 = np.random.randn(hidden_size2, hidden_size1) * 0.01
        self.weights3 = np.random.randn(output_size, hidden_size2) * 0.01
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
    def __init__(self, population_size=250, generations=150, mutation_rate=0.3):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = [NeuralNetwork(204, 120, 80, 34) for _ in range(population_size)]

    def evaluate_fitness_nn(self, nn):
        game = Tetris(10, 20)
        total_fitness = 0
        moves = 0  # Initialize moves as an integer
        StatTracker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        while not game.game_over and moves < 50:
            legal_moves = self.get_legal_moves(game)
            if not legal_moves:
                break
            inputs = self.prepare_inputs(game, legal_moves)
            outputs = nn.forward(inputs)

            # Select the top N outputs where N is the number of legal moves
            top_indices = np.argsort(outputs)[-len(legal_moves):]
            top_outputs = outputs[top_indices]

            # Apply softmax to the selected outputs to get probabilities
            probabilities = softmax(top_outputs)

            # Ensure probabilities have the same size as legal_moves
            if len(probabilities) != len(legal_moves):
                probabilities = np.ones(len(legal_moves)) / len(legal_moves)

            # Choose a move based on the probabilities
            move_index = np.random.choice(len(legal_moves), p=probabilities)
            move = legal_moves[move_index]

            game.current_piece.x += move[0]
            game.current_piece.rotation = (game.current_piece.rotation + move[1]) % len(game.current_piece.shape)
            while game.valid_move(game.current_piece, 0, 1, 0):
                game.current_piece.y += 1
            game.lock_piece(game.current_piece)
            game.update_trackers()
            moves += 1
            total_fitness += game.evaluate_fitness(moves)
            StatTracker = [game.oneCleared, game.twoCleared, game.threeCleared, game.fourCleared, game.holes,
                           game.total_height, game.average_height, game.bumpiness, game.variance, moves]
        return total_fitness, game.binary_grid, StatTracker

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

        # Find the highest row with a 1
        non_empty_rows = [i for i, row in enumerate(binary_grid) if any(row)]
        if non_empty_rows:
            highest_row = max(non_empty_rows)
            start_row = max(0, highest_row - 16)
            end_row = highest_row + 1
        else:
            start_row = max(0, len(binary_grid) - 17)
            end_row = len(binary_grid)

        # Extract the top 10 used rows or the bottom 10 rows
        top_rows = binary_grid[start_row:end_row].flatten()

        features = [game.average_height, game.total_height, game.holes, game.oneCleared, game.twoCleared,
                    game.threeCleared, game.fourCleared, game.bumpiness,game.variance]

        # Add the 5x5 array of the first rotation of the current piece
        first_rotation = np.array(game.current_piece.shape[0]).flatten()
        features.extend(first_rotation)

        return np.concatenate([top_rows, features])

    def train(self):
        no_improvement_generations = 0
        best_fitness = -float('inf')

        for generation in range(self.generations):
            fitness_scores_and_grids_trackers = [self.evaluate_fitness_nn(nn) for nn in self.population]
            fitness_scores = [score for score, _, _ in fitness_scores_and_grids_trackers]
            grids = [grid for _, grid, _ in fitness_scores_and_grids_trackers]
            trackers = [tracker for _, _, tracker in fitness_scores_and_grids_trackers]

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
            print(f'Generation {generation + 1}, Best Fitness: {current_best_fitness}')

            # Render the grid for the best neural network
            best_grid = grids[fitness_scores.index(current_best_fitness)]
            game = Tetris(10, 20)
            game.binary_grid = best_grid
            game.render()

            best_tracker = trackers[fitness_scores.index(current_best_fitness)]
            print(f'One Cleared: {best_tracker[0]}, Two Cleared: {best_tracker[1]}'
                  f', Three Cleared: {best_tracker[2]}, Four Cleared: {best_tracker[3]},'
                  f' Holes: {best_tracker[4]}, Total Height: {best_tracker[5]},'
                  f' Average Height: {best_tracker[6]} Bumpiness: {best_tracker[7]},'
                  f' Variance: {best_tracker[8]} Moves: {best_tracker[9]}')

            # Check for improvement
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            # Early stopping if no improvement over 20 generations
            if no_improvement_generations >= 150:
                print('No improvement for 20 generations, stopping early.')
                break

        best_nn = self.population[0]
        with open('best_nn.pkl', 'wb') as f:
            pickle.dump(best_nn, f)
        print('Training complete, best model saved.')

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size1, parent1.hidden_size2, parent1.output_size)
        # Crossover weights and biases
        child.weights1 = np.where(np.random.rand(*parent1.weights1.shape) < 0.5, parent1.weights1, parent2.weights1)
        child.weights2 = np.where(np.random.rand(*parent1.weights2.shape) < 0.5, parent1.weights2, parent2.weights2)
        child.weights3 = np.where(np.random.rand(*parent1.weights3.shape) < 0.5, parent1.weights3, parent2.weights3)
        child.bias1 = np.where(np.random.rand(*parent1.bias1.shape) < 0.5, parent1.bias1, parent2.bias1)
        child.bias2 = np.where(np.random.rand(*parent1.bias2.shape) < 0.5, parent1.bias2, parent2.bias2)
        child.bias3 = np.where(np.random.rand(*parent1.bias3.shape) < 0.5, parent1.bias3, parent2.bias3)
        return child


if __name__ == "__main__":
    trainer = EvolutionaryTrainer()
    trainer.train()
