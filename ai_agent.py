import copy
import numpy as np

class AIAgent:
    def __init__(self, nn):
        self.nn = nn

    def get_legal_moves(self, game):
        legal_moves = []
        piece = game.current_piece
        for rotation in range(len(piece.shape)):
            for x_offset in range(-game.width // 2 - 1, game.width // 2 + 1):
                simulated_piece = copy.deepcopy(piece)
                simulated_piece.x += x_offset
                simulated_piece.rotation = (simulated_piece.rotation + rotation) % len(simulated_piece.shape)
                if game.valid_move(simulated_piece, 0, 0, 0):
                    legal_moves.append((x_offset, rotation))
        return legal_moves

    def simulate(self, game, move):
        g = game.clone()
        dx, rot = move
        piece = g.current_piece
        piece.x += dx
        piece.rotation = (piece.rotation + rot) % len(piece.shape)
        while g.valid_move(piece, 0, 1, 0):
            piece.y += 1
        g.lock_piece(piece)
        g.update_trackers()
        return g

    def prepare_inputs_1(self, game):
        binary_grid = np.array(game.binary_grid)

        top_rows = binary_grid.flatten()

        features = [game.average_height, game.total_height, game.oneCleared, game.twoCleared, game.threeCleared,
                    game.fourCleared]

        first_rotation = np.array(game.current_piece.shape[0]).flatten()
        features.extend([game.holes, game.bumpiness, game.totalRowFullness])

        return np.concatenate([top_rows, first_rotation, features])

    def prepare_inputs_2(self, game):
        return np.concatenate([game.column_height, np.array([game.current_piece.index])])

    def prepare_inputs(self, game):
        return (self.prepare_inputs_2 if self.nn.input_size <= 12 else self.prepare_inputs_1)(game)

    def find_best_move(self, game):
        legal = self.get_legal_moves(game)
        best_move, best_val = None, -float("inf")
        for mv in legal:
            sim = self.simulate(game, mv)
            inputs = self.prepare_inputs(sim)
            val = float(self.nn.forward(inputs)[0])
            if val > best_val:
                best_move, best_val = mv, val
        return best_move