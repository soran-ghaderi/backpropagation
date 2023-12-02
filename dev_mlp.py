import random
# from micrograd.engine import Value
import numpy as np
import csv

from autodiff import Variable, Adam, Optimizer, DefaultOpt
from loss import Loss, MeanSquaredError


# random.seed(1)
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


    def train(self, xs, ys, epochs=100, verbose=1, early_stopping=None, save_best_model=None):
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

class Neuron:

    def __init__(self, activation_function=None):
        # self.w = [Variable(random.uniform(-1, 1)) for _ in range(nin)]
        self.w = None
        self.b = Variable(random.uniform(-1, 1))
        self.activation_function = activation_function
        # self.nonlin = use_activation
    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x, *args, **kwargs):
        if self.w == None:
            self.w = [Variable(random.uniform(-1, 1)) for _ in
                      range(len(x))]  # infers the inp_no in the runtime instead of asking explicitly
        out = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        if not self.activation_function==None:
            if self.activation_function == 'relu':
                out = out.relu()
            elif self.activation_function=='sigmoid':
                out = out.sigmoid()

        return out

class Dense:
    def __init__(self, units, activation_function=None, **kwargs):
        self.neurons = [Neuron(activation_function, **kwargs) for _ in range(units)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x, *args, **kwargs):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

class Sequential(Model):
    def __init__(self, layers):

        self.layers = layers

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class RMSE(Loss):
    def __init__(self, l2_lambda=0.0):
        self.l2_lambda = l2_lambda
    def __call__(self, y_true, y_pred):
        # Implement mean squared error loss
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        loss = np.mean(np.square(y_true - y_pred)).sqrt()

        return loss

# ===============================================================================================

def load_data():
    global inputs, outputs, data
    file = open('./data/normalized.csv')
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    print(header)
    rows = []
    inputs = []
    outputs = []
    for row in csvreader:
        # load the csv data:
        rows.append(row)
        inputs.append([float(num) for num in row[0:2]])
        outputs.append([float(num) for num in row[2:]])
    data = list(zip(inputs, outputs))
    np.random.shuffle(data)
    inputs, outputs = zip(*data)
    return inputs, outputs

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_data2(file_path='./data/normalized.csv', test_size=0.2, random_state=42):
    # Read the CSV file using pandas
    df = pd.read_csv(file_path)

    # Separate features (inputs) and labels (outputs)
    inputs = df.iloc[:, :2].values
    outputs = df.iloc[:, 2:].values

    # Shuffle the data
    inputs, outputs = shuffle(inputs, outputs, random_state=random_state)

    # Split the data into training and testing sets
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
        inputs, outputs, test_size=test_size, random_state=random_state
    )

    return inputs_train, inputs_test, outputs_train, outputs_test

# Example usage:
inputs_train, inputs_test, outputs_train, outputs_test = load_data2()
print("Inputs Train Shape:", inputs_train.shape, inputs_train)
print("Inputs Test Shape:", inputs_test.shape)
print("Outputs Train Shape:", outputs_train.shape)
print("Outputs Test Shape:", outputs_test.shape)

load_data()
# load_data2()

# train the model:
# ===============================================================================================
activation_function = 'sigmoid'
beta1 = 0.9
beta2 = 0.999
lr = 0.1
clip_threshold = 1.0


model = Sequential([
    Dense(64, activation=activation_function),
    Dense(32, activation=activation_function),
    Dense(16, activation=activation_function),
    Dense(2)  # Output layer, no activation for regression
])
optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, clip_threshold=clip_threshold)
# optimizer = DefaultOpt(learning_rate=0.1)
loss = RMSE()
# loss = CustomMSE()

model.compile(optimizer=optimizer, loss=loss)

model.train(xs=inputs_train, ys=outputs_train, epochs=60, verbose=1,
            early_stopping={'min_delta': 0.001, 'patience': 5},
            save_best_model='best_model_weights.npy')
# pred = model.predict(inputs_test)
evaluation_results = model.evaluate(inputs_test, outputs_test)
evaluation_result_train = model.evaluate(inputs_train, outputs_train)
print("Evaluation Results after training:", evaluation_results, evaluation_result_train)
model.save_model('saved_model_weights.npy')
# pred = model.predict(inputs[-20:])
# print('predicted: ', pred)
# print('true     : ', outputs_test)
# print('sub: ', np.mean(np.array(pred) - np.array(outputs[-20:])))

print(model.predict([[0.9558049704289108,0.6555049379905104]]))
model.load_model('saved_model_weights.npy')
print(model.predict([[0.9558049704289108,0.6555049379905104]]))
evaluation_results = model.evaluate(inputs_test, outputs_test)
print("Evaluation Results after loading best weights:", evaluation_results)
# 0.11828954778867262,0.5061178701108225

