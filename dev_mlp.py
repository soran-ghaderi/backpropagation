# import math
# import numpy as np
# import matplotlib.pyplot as plt
#
# def f(x):
#     return 3*x**2 - 4*x + 5
#
# print(f(3.0))
# xs = np.arange(-5, 5, 0.25)
# ys = f(xs)
#
# # plt.plot(xs, ys)
# # plt.show()
# # ----------------------------
#
# h = 0.000001
# x = 2/3
# y2 = (f(x+h) - f(x))/h
# print(y2)
#
# # ----------------------------
#
# a = 2
# b = -3
# c = 10
# d = a * b + c
# print("d: ", d)
#
# # ----------------------------
# h = 0.00001
# a = 2
# b = -3
# c = 10
# d1 = a * b + c
# c += h
# d2 = a * b + c
# print("d1: ", d1, ' d2: ', d2)
# print('slope: ', (d2-d1)/h)


class Value:
    def __init__(self, value, _children=(), _op='', label=''):
        self.data = value
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, _children=(self, other), _op='+')

    def __mul__(self, other):
        return Value(self.data * other.data, _children=(self, other), _op='*')

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e + c; d.label='d'
f = Value(-2.0, label='f')
L = d * f; L.label='L'

print(d, d._prev, d._op)