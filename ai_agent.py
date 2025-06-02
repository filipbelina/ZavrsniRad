import time
import copy
import numpy as np

class AIAgent:
    def __init__(self, nn):
        self.nn = nn
        self.last_move_time = time.time()

    def copy_game_state(self, game):
        return copy.deepcopy(game)

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
        g = copy.deepcopy(game)
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
        binary = np.array(game.binary_grid).flatten()
        first_rot = np.array(game.current_piece.shape[0]).flatten()
        features = [game.average_height,
                    game.total_height,
                    game.holes,
                    game.bumpiness,
                    game.totalRowFullness]
        return np.concatenate([binary, first_rot, features])

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

    def do_best_move(self, game):
        mv = self.find_best_move(game)
        if mv is None:
            return
        dx, rot = mv
        p = game.current_piece
        p.x += dx
        p.rotation = (p.rotation + rot) % len(p.shape)
        while game.valid_move(p, 0, 1, 0):
            p.y += 1
        game.lock_piece(p)