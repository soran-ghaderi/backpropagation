import random

import numpy as np

from autodiff import Variable


# random.seed(1)
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Variable(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Variable(random.uniform(-1, 1))
        self.nonlin = nonlin

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x, *args, **kwargs):
        # self.w = [Value(random.uniform(-1, 1)) for _ in
        #           range(len(x))]  # infers the inp_no in the runtime instead of asking explicitly
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        out = act.relu()
        # out = act.sigmoid()

        return act.relu() if self.nonlin else act


class Layer(Module):
    def __init__(self, nin, out_no, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(out_no)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x, *args, **kwargs):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out


class MLP(Module):
    def __init__(self, nin, out_nos: list):
        sz = [nin] + out_nos
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(out_nos)-1) for i in range(len(out_nos))]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
def squared_error(y_true, y_pred):
    return sum((ygt - yout) ** 2 for ygt, yout in zip(y_true, y_pred))

def calculate_loss(y_true, y_pred):
    return squared_error(y_true, y_pred) / len(y_true)  # Normalize by the number of samples


xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]

ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP(3, [4, 2, 1])


def train(epochs=1000, lr=0.03):
    for epoch in range(epochs):
        y_pred = [mlp(x) for x in xs]

        loss = calculate_loss(ys, y_pred)
        for p in mlp.parameters():
            p.grad = 0  # Zero out gradients after updating parameters

        loss.backward()
        for p in mlp.parameters():
            # print('before: ', p)
            p.data += -lr * p.grad
            # p.grad = 0  # Zero out gradients after updating parameters
            # print('after: ', p)

        print(f"loss at epoch {epoch}: ", loss.data, [v.data for v in y_pred])


train()
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
