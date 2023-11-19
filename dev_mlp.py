import random

from autodiff import Value
class Neuron:

    def __init__(self, inp_no):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(inp_no)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x, *args, **kwargs):
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        out = act.sigmoid()
        print(act, out)
        return out


x = [2.0, 3.0]
n = Neuron(2)
print("here:", n(x))


