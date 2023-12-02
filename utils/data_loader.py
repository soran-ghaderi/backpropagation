import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DataLoader:
    def __init__(self, file_path='./data/normalized.csv', test_size=0.2, validation_size=None, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.df = None  # Placeholder for the DataFrame

    def load_data(self):
        # Read the CSV file using pandas
        self.df = pd.read_csv(self.file_path)

        # Separate features (inputs) and labels (outputs)
        inputs = self.df.iloc[:, :2].values
        outputs = self.df.iloc[:, 2:].values

        # Shuffle the data
        inputs, outputs = shuffle(inputs, outputs, random_state=self.random_state)

        # Split the data into training and testing sets
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
            inputs, outputs, test_size=self.test_size, random_state=self.random_state
        )

        if not self.validation_size == None:
            # Split the training set into training and validation sets
            inputs_train, inputs_val, outputs_train, outputs_val = train_test_split(
                inputs_train, outputs_train, test_size=self.validation_size, random_state=self.random_state
            )

            return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test

        return inputs_train, inputs_test, outputs_train, outputs_test
