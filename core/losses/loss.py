from abc import ABC

import numpy as np


class Loss(ABC):
    pass


class MeanSquaredError(Loss):
    def __call__(self, y_true, y_pred):
        # loss = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        loss = np.mean(np.square(y_true - y_pred))
        return loss

class SimpleError(Loss):
    def __call__(self, y_true, y_pred):
        # error loss
        # loss = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # loss = np.sum(np.sqrt((y_true - y_pred)**2))
        loss = np.sum(-(y_true - y_pred))
        return loss

class RMSE(Loss):
    def __call__(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        loss = np.mean(np.square(y_true - y_pred)).sqrt()

        return loss
