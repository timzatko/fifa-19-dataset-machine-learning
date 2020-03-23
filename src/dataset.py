import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DatasetForClassification:
    def __init__(self,
                 path,
                 classes):
        data = pd.read_csv(path)

        self.encoders = {}

        for cls in classes:
            le = LabelEncoder()
            data[cls] = le.fit_transform(data[cls])

            self.encoders[cls] = le

        self.y = {}

        for cls in classes:
            self.y[cls] = data[cls].values

        self.X = data.drop(columns=classes)

    def get_labels(self, cls):
        return self.encoders[cls].classes_

    def get_data(self, cls, **kwargs):
        X = self.X
        y = self.y[cls]

        return train_test_split(X, y, test_size=kwargs.get('test_size', 0.2), random_state=42)


# Example
# dataset = DatasetForClassification('../data/fifa_processed_for_cls.csv', ['Position (4)', 'Position (13)'])
# dataset.get_labels('Position (13)')
# X_train, X_test, y_train, y_test = dataset.get_data('Position (13)')


class DatasetForRegression:
    def __init__(self,
                 path,
                 values):
        data = pd.read_csv(path)

        self.y = {}

        for cls in values:
            self.y[cls] = data[cls]

        self.X = data.drop(columns=values)

    def get_data(self, value, **kwargs):
        X = self.X
        y = self.y[value]

        return train_test_split(X, y, test_size=kwargs.get('test_size', 20), random_state=42)
