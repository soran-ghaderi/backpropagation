import pandas as pd
class DataProcessor:
    def __init__(self, file_path):
        self.data = self.read_data(file_path)
        self.min_dict, self.max_dict = self.get_min_max()

    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None)
        return data

    def get_min_max(self):
        min_dict = {}
        max_dict = {}
        for col in self.data.columns:
            min_dict[col] = self.data[col].min()
            max_dict[col] = self.data[col].max()
        return min_dict, max_dict

    def normalize(self, save_path='normalized_data.csv', save=True):
        normalized_data = self.data.copy()
        for col in self.data.columns:
            normalized_data[col] = (self.data[col] - self.min_dict[col]) / (self.max_dict[col] - self.min_dict[col])

        if save:
            normalized_data.to_csv(save_path, index=False, header=False)
            print(f"Normalized data saved to {save_path}")
        return normalized_data

    def denormalize(self, normalized_data, path='denormalized_data.csv', save=True):
        denormalized_data = normalized_data.copy()
        for col in normalized_data.columns:
            denormalized_data[col] = normalized_data[col] * (self.max_dict[col] - self.min_dict[col]) + self.min_dict[col]

        if save:
            denormalized_data.to_csv(path, index=False, header=False)
            print(f"Denormalized data saved to {path}")
        return denormalized_data

class NormalizedDataWrapper:
    def __init__(self, original_data_path):
        self.original_data = self.read_data(original_data_path)
        self.min_dict, self.max_dict = self.get_min_max()

    def read_data(self, file_path):
        data = pd.read_csv(file_path, header=None)
        return data

    def get_min_max(self):
        min_dict = {}
        max_dict = {}
        for col in self.original_data.columns:
            min_dict[col] = self.original_data[col].min()
            max_dict[col] = self.original_data[col].max()
        return min_dict, max_dict

    def normalize_input_pairs(self, vector, column_names=[0, 1]):
        normalized_vector = vector.copy()
        for i, col in enumerate(column_names):
            normalized_vector[i] = (vector[i] - self.min_dict[col]) / (self.max_dict[col] - self.min_dict[col])
        return normalized_vector

    def denormalize_output_pairs(self, normalized_vector, column_names=[2,3]):
        denormalized_vector = normalized_vector.copy()
        for i, col in enumerate(column_names):
            denormalized_vector[i] = normalized_vector[i] * (self.max_dict[col] - self.min_dict[col]) + self.min_dict[col]
        return denormalized_vector


# Example usage:
# file_path = "../data/data.csv"
#
# # Create a DataProcessor instance
# data_processor = DataProcessor(file_path)
#
# # Normalize the data
# normalized_data = data_processor.normalize()
#
# # Create a NormalizedDataWrapper instance
# wrapper = NormalizedDataWrapper(file_path)
# norm = wrapper.normalize_input_pairs([34.09591157410364, 66.12220011209467])
# print('normalized: ', norm)
# # Further operations with the normalized data
# # ...
#
# # When you need to denormalize the data
# denormalized_data = wrapper.denormalize_output_pairs([1.0,0.2400162937283891])
# print('denormalized; ', denormalized_data)
