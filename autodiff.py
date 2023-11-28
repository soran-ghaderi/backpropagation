import math
import numpy as np


class Variable:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Variable(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Variable(sig, (self,), 'sigmoid')
        def _backward():
            self.grad += (sig * (1 - sig)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Variable(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # topological order of all children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


from abc import ABC, abstractmethod
class Optimizer(ABC):
    @abstractmethod
    def step(self, parameters, loss):
        pass
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentums = []
        self.velocities = []
        self.t = 0

    def step(self, parameters, loss):
        self.t += 1

        if not self.momentums:
            self.momentums = [Variable(0) for _ in parameters]
            self.velocities = [Variable(0) for _ in parameters]

        loss.backward()

        for p, m, v in zip(parameters, self.momentums, self.velocities):
            m.data = self.beta1 * m.data + (1 - self.beta1) * p.grad
            v.data = self.beta2 * v.data + (1 - self.beta2) * (p.grad ** 2)

            m_hat = m.data / (1 - self.beta1 ** self.t)
            v_hat = v.data / (1 - self.beta2 ** self.t)

            update = -self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
            p.data += update

            p.grad = 0

        # self.zero_grad()

    def zero_grad(self):
        pass

class MeanSquaredError:
    def __call__(self, y_true, y_pred):
        # Implement mean squared error loss
        loss = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        return loss