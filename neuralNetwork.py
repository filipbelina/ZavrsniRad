import numpy as np

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