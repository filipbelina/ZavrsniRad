import random
import time

class AIAgent:
    def __init__(self):
        self.last_move_time = time.time()

    def get_move(self):
        if time.time() - self.last_move_time >= 1:  # 1 second delay
            self.last_move_time = time.time()
            return random.choice(["left", "right", "drop"])
        return None
