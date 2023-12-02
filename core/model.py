import numpy as np

from loss import MeanSquaredError, Loss
from optimizer import Adam, Optimizer


class Model:

    def __init__(self, sequential):
        self.sequential = sequential
        self.parameters_initialized = False
        self.best_loss = float('inf')
        self.early_stopping_counter = 0


    def compile(self, optimizer='adam', loss='mean_squared_error'):
        if optimizer=='adam':
            self.optimizer = Adam(learning_rate=0.01)
        elif isinstance(optimizer, Optimizer):  # Check if optimizer is an instance of the Optimizer class
            self.optimizer = optimizer
        else:
            raise ValueError("Invalid optimizer. Please provide a valid optimizer or a callable.")

        if loss == 'mean_squared_error':
            self.loss_fn = MeanSquaredError()
        elif isinstance(loss, Loss):  # Check if optimizer is an instance of the Optimizer class
            self.loss_fn = loss
        else:
            raise ValueError("Invalid loss. Please provide a valid loss or a callable.")


    def fit(self, xs, ys, epochs=100, verbose=1, early_stopping=None, save_best_model=None):
        self.best_loss = float('inf')
        print('self.best_loss: ', self.best_loss)
        for epoch in range(epochs):
            y_pred = self.predict(xs)
            loss = self.calculate_loss(ys, y_pred)

            self.optimizer.step(parameters=self.parameters(), loss=loss)

            if verbose == 1:
                print(f"loss at epoch {epoch}: ", loss.data)

            # Early stopping check
            if early_stopping is not None:
                if loss.data < early_stopping['min_delta']:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= early_stopping['patience']:
                        print("Early stopping: Training stopped as the loss did not improve.")
                        break
                else:
                    self.early_stopping_counter = 0

            # Save the best model
            if save_best_model is not None:
                if loss.data < self.best_loss:
                    self.best_loss = loss.data
                    self.save_model(save_best_model)

    def evaluate(self, xs, ys):

        y_pred = self.predict(xs)
        loss = self.calculate_loss(ys, y_pred)

        # Additional evaluation metrics can be added here

        evaluation_metrics = {
            'loss': loss.data
            # Add more metrics as needed
        }

        return evaluation_metrics

    def calculate_loss(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)
    def predict(self, xs):
        # print(self.layers)
        return [self(x) for x in xs]

    def param_init_dummy(self, x=[[0.2, 0.3]]):
        # call the model with a dummy input to initiate parameters
        model.predict(x)
    def parameters(self):
        if not self.parameters_initialized:
            self.param_init_dummy()
            self.parameters_initialized = True

        return [param.data for param in self.sequential.parameters()]

    def save_model(self, file_path='model_weights.npy'):
        model_weights = [param.data for param in self.parameters()]
        np.save(file_path, model_weights)
        print(f"Model weights saved to {file_path}")

    def load_model(self, file_path='model_weights.npy'):
        try:
            model_weights = np.load(file_path, allow_pickle=True)
            print("model_weights: ", self.parameters())
            for param, loaded_weight in zip(self.parameters(), model_weights):
                param.data = loaded_weight
            print(f"Model weights loaded from {file_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")


class Sequential(Model):
    def __init__(self, layers):

        self.layers = layers

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
