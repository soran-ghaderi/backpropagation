import random

from core.autodiff import Variable


class Neuron:

    def __init__(self, activation=None):
        # self.w = [Variable(random.uniform(-1, 1)) for _ in range(nin)]
        self.w = None
        self.b = Variable(random.uniform(-1, 1))
        self.activation_function = activation
        # self.nonlin = use_activation
    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x, *args, **kwargs):
        if self.w == None:
            self.w = [Variable(random.uniform(-1, 1)) for _ in
                      range(len(x))]  # infers the inp_no in the runtime instead of asking explicitly
        out = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        if not self.activation_function==None:
            if self.activation_function == 'relu':
                out = out.relu()
            elif self.activation_function=='sigmoid':
                out = out.sigmoid()

        return out


class Dense:
    def __init__(self, units, activation=None, **kwargs):
        self.neurons = [Neuron(activation, **kwargs) for _ in range(units)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x, *args, **kwargs):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
