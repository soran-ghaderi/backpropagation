# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, inputs, gradient):
        self.weights -= 0.01 * np.dot(gradient, inputs)
        self.bias -= 0.01 * gradient
        return gradient * self.weights


class Dense:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, inputs, gradients):
        return [neuron.backward(inputs, gradient) for neuron, gradient in zip(self.neurons, gradients)]


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, inputs, gradients):
        for layer in reversed(self.layers):
            gradients = layer.backward(inputs, gradients)
        return gradients


# Example usage
dense1 = Dense(2, 3)
dense2 = Dense(3, 1)
network = Network([dense1, dense2])

# Training example
for _ in range(1000):
    inputs = np.random.rand(2)
    target = np.random.rand()
    output = network.forward(inputs)
    loss = (output[0] - target) ** 2
    gradients = [2 * (output[0] - target)]
    network.backward(inputs, gradients)

# Testing
test_input = np.array([0.5, 0.5])
print(network.forward(test_input))
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
