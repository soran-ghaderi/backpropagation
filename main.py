import json
from itertools import product

from core.layers import Dense
from core.model import Sequential, HyperparameterTuner
from core.optimizers import Adam
from core.losses import RMSE, MeanSquaredError, SimpleError
from utils import DataLoader, NormalizedDataWrapper, DataProcessor

# ================================ Loading data ===============================================================
original_file_path = 'data/ce889_dataCollection.csv'
normalized_output_path = 'data/normalized_data.csv'
data_processor = DataProcessor(original_file_path)
normalized_data = data_processor.normalize(save_path=normalized_output_path)
data_wrapper = NormalizedDataWrapper(original_file_path)
data_loader = DataLoader(file_path=normalized_output_path, validation_size=0.1)
# Load data
data_loader.load_data(1000)
# Split data into training, validation, and testing sets
inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test = data_loader.split_data()

# Example usage:
# inputs_train, inputs_test, outputs_train, outputs_test = load_data()
print("Inputs Train Shape:", inputs_train.shape)
print("Inputs Val Shape:", inputs_val.shape)
print("Inputs Test Shape:", inputs_test.shape)
print("Outputs Train Shape:", outputs_train.shape)
print("Outputs Val Shape:", outputs_val.shape)
print("Outputs Test Shape:", outputs_test.shape)

# optimizer = SimpleSGD(learning_rate=0.1)

# train the model:
# ===============================================================================================
beta1 = 0.9
beta2 = 0.999
lr = 0.1
clip_threshold = 1.0

activation_function = 'sigmoid'
# model = Sequential([
#     Dense(2, activation=activation_function),
#     Dense(32, activation=activation_function),
#     Dense(2)  # No activation for regression
# ])

loss2 = RMSE()
loss = MeanSquaredError()
loss3 = SimpleError()

optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, clip_threshold=clip_threshold)
# model.compile(optimizer=optimizer, loss=loss)
#
# model.fit(inputs=inputs_train, outputs=outputs_train, val_inputs=inputs_val, val_outputs=outputs_val, batch_size=1, epochs=50, verbose=1,
#           early_stopping={'min_delta': 0.001, 'patience': 5},
#           save_best_model='./weights/best_model_weights.npy')

# pred = model.predict(inputs_test)
# evaluation_results = model.evaluate(inputs_test, outputs_test)
# print("Evaluation Results after training:", evaluation_results)

# evaluation_result_train = model.evaluate(inputs_train, outputs_train)
# model.save_model('./weights/saved_model_weights.npy')
#
# model.load_model('./weights/best_model_weights.npy')
# print('predicted: ', model.predict([[0.5455777623703603,0.7088149539113109]]))
# print(model.predict([[0.9558049704289108,0.6555049379905104]]))
# evaluation_results = model.evaluate(inputs_test, outputs_test, loss=loss)
# print("Evaluation Results after loading best weights:", evaluation_results)
# 0.11828954778867262,0.5061178701108225

# tuning
hidden_neurons_list = [4, 8, 16]
lr_list = [0.001, 0.01]
beta1_list = [0.7, 0.9]

tuner = HyperparameterTuner(hidden_neurons_list=hidden_neurons_list, lr_list=lr_list, beta1_list=beta1_list)
# best_hyperparameters = tuner.grid_search(inputs_train, outputs_train, inputs_val, outputs_val)

# tuner.visualize_results()


# load more data:
data_loader.load_data()
#
# Split data into training, validation, and testing sets
inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test = data_loader.split_data()

# best_model = Sequential([
#     Dense(2, activation='sigmoid'),
#     Dense(best_hyperparameters['hidden_neurons'], activation='sigmoid'),
#     Dense(2)
# ])

best_model = Sequential([
    Dense(2, activation='sigmoid'),
    Dense(10, activation='sigmoid'),
    Dense(2)
])
#
# best_optimizer = Adam(learning_rate=best_hyperparameters['lr'],
#                       beta1=best_hyperparameters['beta1'],
#                       beta2=0.999,
#                       clip_threshold=1.0)

best_optimizer = Adam(learning_rate=0.01,
                      beta1=0.7,
                      beta2=0.999,
                      clip_threshold=1.0)

best_model.compile(optimizer=best_optimizer, loss=MeanSquaredError())
training_loss, validation_loss = best_model.fit(inputs=inputs_train, outputs=outputs_train, val_inputs=inputs_val, val_outputs=outputs_val, batch_size=256, epochs=10, verbose=1,
               early_stopping={'min_delta': 0.001, 'patience': 3},
               save_best_model='./weights/best_model_weights.npy')

def write_list(a_list, file_name="losses.json"):
    print("Started writing list data into a json file")
    with open(file_name, "w") as fp:
        json.dump(a_list, fp)
        print("Done writing JSON data into .json file")

final_evaluation_results = best_model.evaluate(inputs_test, outputs_test, loss=MeanSquaredError())
print("\nFinal Evaluation Results on Test Set:", final_evaluation_results)
# print("\nBest Hyperparameters combination:", best_hyperparameters)
write_list(training_loss, 'training_loss.json')
write_list(validation_loss, 'validation_loss.json')