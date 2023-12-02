from abc import ABC


class Loss(ABC):
    pass


class MeanSquaredError(Loss):
    def __call__(self, y_true, y_pred):
        # Implement mean squared error loss
        loss = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        return loss
