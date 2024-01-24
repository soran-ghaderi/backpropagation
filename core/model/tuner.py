from itertools import product

import numpy as np

from core.layers import Dense
from core.losses import SimpleError, MeanSquaredError
from core.model import Sequential
from core.optimizers import Adam

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HyperparameterTuner:
    def __init__(self, hidden_neurons_list, lr_list, beta1_list):
        self.hidden_neurons_list = hidden_neurons_list
        self.lr_list = lr_list
        self.beta1_list = beta1_list
        self.best_hyperparameters = None
        self.best_evaluation_loss = float('inf')
        self.grid_results = []
        self.grid_results_dict = {}

    def grid_search(self, inputs_train, outputs_train, inputs_val, outputs_val, activation_function='sigmoid',
                    beta2=0.999, clip_threshold=1.0, epochs=5, batch_size=128, verbose=1):
        max_row, max_col = 3, 3
        fig, axs = plt.subplots(max_row, max_col)
        row, col = 0, 0
        for hidden_neurons, lr, beta1 in product(self.hidden_neurons_list, self.lr_list, self.beta1_list):
            print(f"\nTraining with hidden neurons: {hidden_neurons}, lr: {lr}, beta1: {beta1}")

            model = Sequential([
                Dense(2, activation=activation_function),
                Dense(hidden_neurons, activation=activation_function),
                Dense(2)
            ])

            optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, clip_threshold=clip_threshold)
            loss_fn = MeanSquaredError()

            model.compile(optimizer=optimizer, loss=loss_fn)

            training_loss, validation_loss = model.fit(inputs=inputs_train, outputs=outputs_train, val_inputs=inputs_val, val_outputs=outputs_val,
                      batch_size=batch_size, epochs=epochs, verbose=verbose,
                      early_stopping={'min_delta': 0.001, 'patience': 5})

            self.grid_results_dict[(hidden_neurons, lr, beta1)] = [training_loss, validation_loss]

            if row <= max_row-1 and col <=max_col-1:
                axs[row, col].plot(np.linspace(0, len(training_loss), len(training_loss)), training_loss)
                axs[row, col].plot(np.linspace(0, len(validation_loss), len(validation_loss)), validation_loss)
                axs[row, col].set_title(f"{row}-{col}")
            if row < max_row-1:
                row += 1
            else:
                if col < max_col-1:
                    row = 0
                    col += 1
                else:
                    pass


            evaluation_results = model.evaluate(inputs_val, outputs_val)

            print(f"Evaluation Results for current hyperparameters: {evaluation_results}")

            if evaluation_results['loss'] < self.best_evaluation_loss:
                self.best_evaluation_loss = evaluation_results['loss']
                self.best_hyperparameters = {'hidden_neurons': hidden_neurons, 'lr': lr, 'beta1': beta1}

            # Save results for visualization
            self.grid_results.append((hidden_neurons, lr, beta1, evaluation_results['loss']))

        for ax in axs.flat:
            ax.set(xlabel='Epochs', ylabel='Loss')
        plt.show()

        print("\nBest hyperparameters:", self.best_hyperparameters)
        return self.best_hyperparameters

    def visualize_results(self):
        # Choose two hyperparameters to fix
        fixed_param1 = self.grid_results[0][0]
        fixed_param2 = self.grid_results[0][1]

        fig, axes = plt.subplots(nrows=len(self.grid_results), ncols=1, figsize=(8, 6 * len(self.grid_results)))

        for i, (param1, param2, varying_param, losses) in enumerate(self.grid_results):
            # Plot the results
            axes[i].plot(varying_param, losses, marker='o', linestyle='-', label=f'{param1}: {fixed_param1}, {param2}: {fixed_param2}')
            axes[i].set_xlabel('Varying Parameter')
            axes[i].set_ylabel('Validation Loss')
            axes[i].legend()

        plt.suptitle(f'Grid Search Results for Fixed Parameters ({param1}: {fixed_param1}, {param2}: {fixed_param2})')
        plt.show()
