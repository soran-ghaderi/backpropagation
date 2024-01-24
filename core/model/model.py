import os

import numpy as np
from tqdm import tqdm as tqdm_progress

from core.losses.loss import MeanSquaredError, Loss
from core.optimizers.optimizer import Adam, Optimizer


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

    def fit(self, inputs, outputs, val_inputs=None, val_outputs=None, batch_size=32, epochs=10, verbose=1, early_stopping=None, save_best_model=None):
        num_samples = len(inputs)
        num_batches = (num_samples // batch_size) if (num_samples // batch_size)>0 else 1

        self.best_loss = float('inf')
        self.training_loss = []
        self.validation_loss = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            with tqdm_progress(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = (batch_idx + 1) * batch_size
                    x_batch = inputs[start_idx:end_idx]
                    y_batch = outputs[start_idx:end_idx]

                    y_pred = self.predict(x_batch)
                    loss = self.calculate_loss(y_batch, y_pred)
                    # print('loss at training: ', loss)

                    self.optimizer.step(parameters=self.parameters(), loss=loss)

                    epoch_loss += loss.data
                    pbar.set_postfix(loss=f"{loss.data:.4f}")
                    pbar.update(1)

                epoch_loss /= num_batches

                val_loss = 0.0
                if val_inputs is not None and val_outputs is not None:
                    val_loss = self.evaluate(val_inputs, val_outputs)['loss']
                    pbar.set_postfix(train_loss=f"{epoch_loss:.4f}", val_loss=f"{val_loss:.4f}")
                else:
                    pbar.set_postfix(train_loss=f"{epoch_loss:.4f}")
            self.validation_loss.append(val_loss)
            self.training_loss.append(epoch_loss)
            # if verbose == 1:
                # print(f"loss at epoch {epoch}: ", loss.data)
                # print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")


            # Early stopping check
            if early_stopping is not None:
                if loss.data <= early_stopping['min_delta']:
                    break

                if loss.data >= self.best_loss - early_stopping['min_delta']:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= early_stopping['patience']:
                        print("Early stopping: Training stopped as the loss did not improve.")
                        return self.training_loss, self.validation_loss
                else:
                    self.early_stopping_counter = 0

            # Save the best model
            if save_best_model is not None:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(save_best_model)

        return self.training_loss, self.validation_loss

    # def evaluate(self, xs, ys, loss=None):
    #     if not loss == None:
    #         self.loss_fn = loss
    #     y_pred = self.predict(xs)
    #     loss = self.calculate_loss(ys, y_pred)
    #
    #     # Additional evaluation metrics can be added here
    #
    #     evaluation_metrics = {
    #         'loss': loss.data
    #         # Add more metrics as needed
    #     }
    #
    #     return evaluation_metrics

    def evaluate(self, xs, ys, loss=None, batch_size=128):
        if not loss == None:
            self.loss_fn = loss

        num_samples = len(xs)
        num_batches = (num_samples // batch_size) if (num_samples // batch_size) > 0 else 1

        total_loss = 0.0

        with tqdm_progress(total=num_batches, desc="Evaluation", unit="batch") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                x_batch = xs[start_idx:end_idx]
                y_batch = ys[start_idx:end_idx]

                y_pred = self.predict(x_batch)
                batch_loss = self.calculate_loss(y_batch, y_pred)

                total_loss += batch_loss.data
                pbar.set_postfix(loss=f"{batch_loss.data:.4f}")
                pbar.update(1)

        total_loss /= num_batches

        # Additional evaluation metrics can be added here

        evaluation_metrics = {
            'loss': total_loss
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
        self.predict(x)
    def parameters(self):
        if not self.parameters_initialized:
            self.param_init_dummy()
            self.parameters_initialized = True

        return [param.data for param in self.sequential.parameters()]

    def save_model(self, file_path='model_weights.npy'):
        model_weights = [param.data for param in self.parameters()]


        # create directory if does not exist
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

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
            try:
                self.param_init_dummy()
                model_weights = np.load(file_path, allow_pickle=True)
                print("model_weights: ", self.parameters())
                for param, loaded_weight in zip(self.parameters(), model_weights):
                    param.data = loaded_weight
                print(f"Model weights loaded from {file_path}")
            except:
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
