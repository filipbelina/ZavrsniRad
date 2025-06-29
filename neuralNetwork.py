import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, output_size=1):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.output_size = output_size
        self.weights1 = np.random.randn(self.hidden_size1, input_size) * np.sqrt(1 / input_size)
        self.weights2 = np.random.randn(output_size, self.hidden_size1) * np.sqrt(1 / self.hidden_size1)
        self.bias1 = np.zeros((self.hidden_size1, 1))
        self.bias2 = np.zeros((output_size, 1))

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        a1 = np.tanh(self.weights1 @ x + self.bias1)
        z3 = self.weights2 @ a1 + self.bias2
        return z3.flatten()

    def mutate(self, rate=0.3):
        def nudge(w, s):
            layer_std = np.std(w)
            return w + np.random.randn(*w.shape) * s * max(layer_std, 1e-3)

        self.weights1 = nudge(self.weights1, rate)
        self.weights2 = nudge(self.weights2, rate)
        self.bias1    = nudge(self.bias1, rate)
        self.bias2    = nudge(self.bias2, rate)