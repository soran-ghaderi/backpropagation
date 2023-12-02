# from micrograd.engine import Value

from core.layers import Dense
from core.model import Sequential
from core.optimizers import Adam
from core.losses import RMSE
from utils import DataLoader

# ===============================================================================================


data_loader = DataLoader()
inputs_train, inputs_test, outputs_train, outputs_test = data_loader.load_data()

# Example usage:
# inputs_train, inputs_test, outputs_train, outputs_test = load_data()
print("Inputs Train Shape:", inputs_train.shape, inputs_train)
print("Inputs Test Shape:", inputs_test.shape)
print("Outputs Train Shape:", outputs_train.shape)
print("Outputs Test Shape:", outputs_test.shape)


# train the model:
# ===============================================================================================
activation_function = 'sigmoid'
beta1 = 0.9
beta2 = 0.999
lr = 0.1
clip_threshold = 1.0


model = Sequential([
    Dense(16, activation=activation_function),
    Dense(8, activation=activation_function),
    Dense(2)  # Output layer, no activation for regression
])
optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, clip_threshold=clip_threshold)
# optimizer = SimpleSGD(learning_rate=0.1)
loss = RMSE()
# loss = CustomMSE()

model.compile(optimizer=optimizer, loss=loss)

model.fit(xs=inputs_train, ys=outputs_train, epochs=60, verbose=1,
          early_stopping={'min_delta': 0.001, 'patience': 5},
          save_best_model='./weights/best_model_weights.npy')
# pred = model.predict(inputs_test)
evaluation_results = model.evaluate(inputs_test, outputs_test)
evaluation_result_train = model.evaluate(inputs_train, outputs_train)
print("Evaluation Results after training:", evaluation_results, evaluation_result_train)
model.save_model('./weights/saved_model_weights.npy')

print(model.predict([[0.9558049704289108,0.6555049379905104]]))
model.load_model('./weights/saved_model_weights.npy')
print(model.predict([[0.9558049704289108,0.6555049379905104]]))
evaluation_results = model.evaluate(inputs_test, outputs_test)
print("Evaluation Results after loading best weights:", evaluation_results)
# 0.11828954778867262,0.5061178701108225

