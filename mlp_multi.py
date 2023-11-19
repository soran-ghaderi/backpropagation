import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        self.biases = [np.random.randn(1, size) for size in self.layer_sizes[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        layer_output = X
        for i in range(len(self.layer_sizes) - 1):
            layer_output = self.activations[i](np.dot(layer_output, self.weights[i]) + self.biases[i])
        return layer_output

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, X, y, output, learning_rate):
        errors = [None] * len(self.layer_sizes)
        errors[-1] = y - output

        for i in range(len(self.layer_sizes) - 2, -1, -1):
            errors[i] = np.dot(errors[i + 1], self.weights[i]) * self.sigmoid_derivative(output)

            grad_weights = np.dot(errors[i + 1].T,
                                  self.activations[i](np.dot(self.weights[i], X.T) + self.biases[i].T)).T
            grad_biases = np.sum(errors[i + 1], axis=0, keepdims=True)

            self.weights[i] += learning_rate * grad_weights
            self.biases[i] += learning_rate * grad_biases


    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

# Example usage
layer_sizes = [2, 3, 4, 1]  # Example layer sizes: input, hidden1, hidden2, output
activations = [np.tanh, np.tanh, lambda x: x]  # Example activation functions
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(layer_sizes, activations)
nn.train(X, y, learning_rate=0.1, epochs=1000)

# Test the trained model
print("Final predictions after training:")
print(nn.forward(X))