import time
import copy

class AIAgent:
    def __init__(self):
        self.last_move_time = time.time()

    def copy_game_state(self, game):
        """Create a deep copy of the game state"""
        return copy.deepcopy(game)

    def simulate_move(self, game, piece, x_offset, rotation):
        """Simulate a move and return the resulting fitness"""
        simulated_piece = copy.deepcopy(piece)  # Copy the piece separately
        simulated_piece.x += x_offset
        simulated_piece.rotation = (simulated_piece.rotation + rotation) % len(simulated_piece.shape)

        if game.valid_move(simulated_piece, 0, 0, 0):
            while game.valid_move(simulated_piece, 0, 1, 0):
                simulated_piece.y += 1
            game.lock_piece(simulated_piece)
            game.update_trackers()
            return game.evaluate_fitness()
        return float('-inf')

    def find_best_move(self, game):
        best_move = None
        best_fitness = float('-inf')
        piece = game.current_piece

        for rotation in range(len(piece.shape)):
            for x_offset in range(-game.width // 2 - 1, game.width // 2 + 1):
                simulated_game = self.copy_game_state(game)
                simulated_piece = simulated_game.current_piece
                fitness = self.simulate_move(simulated_game, simulated_piece, x_offset, rotation)
                print(f"best_fitness: {best_fitness}, fitness: {fitness}, x_offset: {x_offset}, rotation: {rotation}")
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_move = (x_offset, rotation)
        print(f"Best move: {best_move}, fitness: {best_fitness}")
        return best_move

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