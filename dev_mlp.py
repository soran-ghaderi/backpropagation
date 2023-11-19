import random

from autodiff import Value
class Neuron:

    def __init__(self, inp_no):
        self.w = None
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x, *args, **kwargs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(len(x))]
        act = sum((wi*xi for wi, xi in zip(self.w, x)), start=self.b)
        out = act.sigmoid()
        return out


x = [2.0, 3.0, 4.0]
n = Neuron(2)
print("here:", n(x))


