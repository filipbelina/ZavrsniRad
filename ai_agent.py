import time
import copy
import numpy as np

class AIAgent:
    def __init__(self, nn):
        self.nn = nn
        self.last_move_time = time.time()

    def copy_game_state(self, game):
        """Create a deep copy of the game state"""
        return copy.deepcopy(game)

    def prepare_inputs(self, game, moves):
        binary_grid = np.array(game.binary_grid)

        top_rows = binary_grid.flatten()

        features = [game.average_height, game.total_height]

        # Add the 5x5 array of the first rotation of the current piece
        first_rotation = np.array(game.current_piece.shape[0]).flatten()
        features.extend([game.total_height, game.average_height, game.holes, game.bumpiness, game.totalRowFullness])

        return np.concatenate([top_rows, first_rotation, features])

    def get_legal_moves(self, game):
        """Get all legal moves for the current piece"""
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

    def select_move(self, outputs, legal_moves):
        sorted_indices = np.argsort(outputs)[::-1]
        move_index = None
        for idx in sorted_indices:
            if idx < len(legal_moves):
                move_index = idx
                break
        if move_index is None:
            move_index = sorted_indices[0]
        return move_index

    def find_best_move(self, game):
        legal_moves = self.get_legal_moves(game)
        if not legal_moves:
            return None
        inputs = self.prepare_inputs(game, legal_moves)
        outputs = self.nn.forward(inputs)
        move_index = self.select_move(outputs, legal_moves)
        return legal_moves[move_index]

    def do_best_move(self, game):
        best_move = self.find_best_move(game)
        if best_move:
            x_offset, rotation = best_move
            piece = game.current_piece

            # Apply the rotation
            piece.rotation = (piece.rotation + rotation) % len(piece.shape)

            # Apply the horizontal movement
            piece.x += x_offset

            # Drop the piece
            while game.valid_move(piece, 0, 1, 0):
                piece.y += 1
            time.sleep(0.5)
            game.lock_piece(piece)