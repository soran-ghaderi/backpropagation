import math
from abc import ABC, abstractmethod

import numpy as np

from core.autodiff import Variable


class Optimizer(ABC):
    @abstractmethod
    def step(self, parameters, loss):
        pass


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_threshold=1.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentums = []
        self.velocities = []
        self.t = 0
        self.clip_threshold = clip_threshold

    def step(self, parameters, loss):
        self.t += 1

        if not self.momentums:
            self.momentums = [Variable(0) for _ in parameters]
            self.velocities = [Variable(0) for _ in parameters]

        loss.backward()
        for p, m, v in zip(parameters, self.momentums, self.velocities):
            p.grad = np.clip(p.grad, -self.clip_threshold, self.clip_threshold)

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


class SimpleSGD(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, parameters, loss):

        loss.backward()

        for p in parameters:
            # print('before: ', p)
            p.data += -self.learning_rate * p.grad
            p.grad = 0  # Zero out gradients after updating parameters
