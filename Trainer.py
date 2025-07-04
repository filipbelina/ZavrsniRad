from neuralNetwork import NeuralNetwork
import numpy as np

class EvolutionaryTrainer:
    def __init__(self, population_size=100, generations=100, mutation_rate=0.3, fitness_function=3, training_algorithm=1, evaluations=5000):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.training_algorithm = training_algorithm
        if self.training_algorithm == 1 or self.training_algorithm == 2:
            self.population = [NeuralNetwork(234, 8, 1) for _ in range(population_size)]
        else:
            self.population = [NeuralNetwork(21, 16,1) for _ in range(population_size)]
        self.current_game_state = None
        self.best_neural_network = None
        self.evaluations = evaluations

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

        features = [game.average_height, game.total_height, game.oneCleared, game.twoCleared, game.threeCleared, game.fourCleared]

        first_rotation = np.array(game.current_piece.shape[0]).flatten()
        features.extend([game.holes,game.bumpiness,game.totalRowFullness])

        return np.concatenate([top_rows, first_rotation, features])

    def prepare_inputs2(self, game):
        features = [game.average_height, game.total_height]
        features.extend([game.holes, game.bumpiness, game.totalRowFullness, game.score, game.oneCleared, game.twoCleared, game.threeCleared, game.fourCleared])
        return np.concatenate([game.column_height, np.array([game.current_piece.index]), features])


    def crossover(self, parent1, parent2):
        child = NeuralNetwork(parent1.input_size, parent1.hidden_size1, parent1.output_size)
        def blend(a, b):
            alpha = np.random.rand(*a.shape)
            return alpha * a + (1 - alpha) * b

        child.weights1 = blend(parent1.weights1, parent2.weights1)
        child.weights2 = blend(parent1.weights2, parent2.weights2)
        child.bias1    = blend(parent1.bias1,  parent2.bias1)
        child.bias2    = blend(parent1.bias2,  parent2.bias2)
        return child