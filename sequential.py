import numpy as np

class Sequential:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.random.randn(1, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.random.randn(1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = self.sigmoid(self.z2)
        return output

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, X, y, output, learning_rate):
        # Backward propagation
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        a1_error = np.dot(output_delta, self.W2.T)
        a1_delta = a1_error * self.sigmoid_derivative(self.a1)

        self.W2 += np.dot(self.a1.T, output_delta) * learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.W1 += np.dot(X.T, a1_delta) * learning_rate
        self.b1 += np.sum(a1_delta, axis=0) * learning_rate

    def train(self, X, y, lr, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, lr)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = Sequential(2, 2, 2)
print("before training:")
print(mlp.forward(X))
mlp.train(X, y, lr=0.1, epochs=1000)

print("after training:")
print(mlp.forward(X))


