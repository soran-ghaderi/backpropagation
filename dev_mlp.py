import random

import numpy as np

from autodiff import Variable, Adam, MeanSquaredError, Optimizer

# random.seed(1)
class Model:

    def __init__(self, sequential):
        self.sequential = sequential

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        if optimizer=='adam':
            self.optimizer = Adam(learning_rate=0.01)
        elif isinstance(optimizer, Optimizer):  # Check if optimizer is an instance of the Optimizer class
            self.optimizer = optimizer
        else:
            raise ValueError("Invalid optimizer. Please provide a valid optimizer or a callable.")

        if loss == 'mean_squared_error':
            self.loss_fn = MeanSquaredError()

    def train(self, xs, ys, epochs=100, lr=0.03):
        for epoch in range(epochs):
            y_pred = self.predict(xs)
            loss = self.calculate_loss(ys, y_pred)
            self.optimizer.step(parameters=self.parameters(), loss=loss)
            # loss.backward()
            # #
            # for p in self.parameters():
            #     p.data += -lr * p.grad
            #     p.grad = 0

            print(f"loss at epoch {epoch}: ", loss.data, [v.data for v in y_pred])

    def calculate_loss(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)
    def predict(self, xs):
        # print(self.layers)
        return [self(x) for x in xs]
    def parameters(self):
        return []



class Neuron:

    def __init__(self, nonlin=True):
        # self.w = [Variable(random.uniform(-1, 1)) for _ in range(nin)]
        self.w = None

        self.b = Variable(random.uniform(-1, 1))
        self.nonlin = nonlin

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x, *args, **kwargs):
        if self.w == None:
            self.w = [Variable(random.uniform(-1, 1)) for _ in
                      range(len(x))]  # infers the inp_no in the runtime instead of asking explicitly
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        out = act.relu()
        # out = act.sigmoid()

        return act.relu() if self.nonlin else act


class Dense:
    def __init__(self, out_no, **kwargs):
        self.neurons = [Neuron(**kwargs) for _ in range(out_no)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x, *args, **kwargs):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

class MLP:
    def __init__(self, nin, out_nos: list):
        self.layers = [Dense(out_nos[i], nonlin=i != len(out_nos) - 1) for i in range(len(out_nos))]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

class Sequential(Model):
    def __init__(self, layers):

        self.layers = layers

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# def squared_error(y_true, y_pred):
#     return sum((ygt - yout) ** 2 for ygt, yout in zip(y_true, y_pred))
#
# def calculate_loss(y_true, y_pred):
#     return squared_error(y_true, y_pred) / len(y_true)  # Normalize by the number of samples


xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]

ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP(3, [4, 2, 1])
model = Sequential([
    Dense(4),
    Dense(2),
    Dense(1, nonlin=False)
])


# def train(epochs=100, lr=0.03):
#     for epoch in range(epochs):
#         y_pred = [mlp(x) for x in xs]
#
#         loss = calculate_loss(ys, y_pred)
#         # for p in mlp.parameters():
#         #     p.grad = 0  # Zero out gradients after updating parameters
#
#         loss.backward()
#         for p in mlp.parameters():
#             # print('before: ', p)
#             p.data += -lr * p.grad
#             p.grad = 0  # Zero out gradients after updating parameters
#             # print('after: ', p)
#
#         print(f"loss at epoch {epoch}: ", loss.data, [v.data for v in y_pred])


# train()
optimizer = Adam(learning_rate=0.1)
model.compile(optimizer=optimizer)
model.train(xs, ys, epochs=1000, lr=0.01)

# y_pred = [mlp(x) for x in xs]
#
# # loss = sum((ygt - yout)**2 for ygt, yout in zip(ys, y_pred))
# loss = calculate_loss(ys, y_pred)
#
# print('y_pred_1: ', y_pred)
# # print(ys, y_pred)
# # print(squared_error(ys, y_pred), loss)
# # print('parameters: ', mlp.parameters(), sep=']')
# print('loss: ', loss)
# loss.backward()
# # print(mlp.layers[0].neurons[0].w)
# lr = 0.05
# for p in mlp.parameters():
#     # print('before: ', p.data)
#     p.data += -lr * p.grad
#     # print('after; ', p.data)
#
# y_pred = [mlp(x) for x in xs]
# print('y_pred: ', y_pred)
# print('loss: ', loss)
# loss = calculate_loss(ys, y_pred)
# print('loss: ', loss)
