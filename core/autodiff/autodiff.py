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

    def sqrt(self):
        sqrt_val = np.sqrt(self.data)
        out = Variable(sqrt_val, (self,), 'sqrt')

        def _backward():
            self.grad += 0.5 * (1 / sqrt_val) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # topological order of all children in the graph
        topo = []
        visited = set()
        stack = [self]

        while stack:
            current = stack[-1]

            if current not in visited:
                visited.add(current)
                for child in current._prev:
                    stack.append(child)
            else:
                topo.append(stack.pop())

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
        if other == 0:
            raise ZeroDivisionError("Division by zero is not allowed")
        try:
            res = self * float(other) ** -1
        except Exception as e:
            print(other, e)
        return res

    def __rtruediv__(self, other):  # other / self
        if self == 0:
            raise ZeroDivisionError("Division by zero is not allowed")
        try:
            res = other * self ** -1
        except Exception as e:
            print(self, e)

        return res

    def __repr__(self):
        # return f"Variable(data={self.data}, grad={self.grad})"
        return f"{self.data}"


