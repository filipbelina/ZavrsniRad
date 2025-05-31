from neuralNetwork import NeuralNetwork
import numpy as np

class EvolutionaryTrainer:
    def __init__(self, population_size=10000, generations=1000, mutation_rate=0.3, fitness_function=1, training_algorithm=1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_moves = self.generations * 3
        self.fitness_function = fitness_function
        self.training_algorithm = training_algorithm
        if self.training_algorithm == 1 or self.training_algorithm == 2:
            self.population = [NeuralNetwork(230, 120, 80, 34) for _ in range(population_size)]
        else:
            self.population = [NeuralNetwork(11, 5, 3, 34) for _ in range(population_size)]
        self.current_game_state = None

    def get_legal_moves(self, game):
        piece = game.current_piece
        moves = []
        for rotation in range(len(piece.shape)):
            for x_offset in range(-game.width // 2 - 1, game.width // 2 + 1):
                if game.valid_move(piece, x_offset, 0, rotation):
                    moves.append((x_offset, rotation))
        return moves

    def prepare_inputs1(self, game):
        binary_grid = np.array(game.binary_grid)

        top_rows = binary_grid.flatten()

        features = [game.average_height, game.total_height]

        first_rotation = np.array(game.current_piece.shape[0]).flatten()
        features.extend([game.holes,game.bumpiness,game.totalRowFullness])

        return np.concatenate([top_rows, first_rotation, features])

    def prepare_inputs2(self, game):
        return np.concatenate([game.column_height, np.array([game.current_piece.index])])


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