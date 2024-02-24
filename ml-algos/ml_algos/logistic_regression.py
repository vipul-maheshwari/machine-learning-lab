import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression():
    def __init__(self, X, y, learning_rate=0.001, num_epochs=1000):
        self.X = X
        self.y = y.reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.num_training_examples, self.num_features = None, None

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.weights = None
        self.bias = None

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.num_training_examples, self.num_features = self.X_train.shape

        # Standardize the data for mean 0 and standard deviation 1 (It helps in exponential overflow if the feature values are very large)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)


    def initialize_variables(self):
        self.weights = np.random.random(self.num_features).reshape(-1, 1)
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_pass(self):
        linear_calculation = np.dot(self.X_train, self.weights) + self.bias
        predictions = self.sigmoid(linear_calculation)
        return predictions

    def compute_gradients(self, Y_predictions):
        gradient_weights = (1 / self.num_training_examples) * np.dot(np.transpose(self.X_train), (Y_predictions - self.y_train))
        gradient_bias = (1 / self.num_training_examples) * np.sum(Y_predictions - self.y_train)
        return gradient_weights, gradient_bias

    def update_parameters(self, gradient_weights, gradient_bias):
        self.weights -= self.learning_rate * gradient_weights
        self.bias -= self.learning_rate * gradient_bias

    def train(self):
        for epoch in range(self.num_epochs):
            predictions = self.forward_pass()
            gradient_weights, gradient_bias = self.compute_gradients(predictions)
            self.update_parameters(gradient_weights, gradient_bias)

    def test(self):
        linear_calculation = np.dot(self.X_test, self.weights) + self.bias
        predictions = self.sigmoid(linear_calculation)
        class_labels = [0 if pred < 0.5 else 1 for pred in predictions]
        return class_labels

    def accuracy(self):
        class_labels = self.test()
        accuracy = np.sum(class_labels == self.y_test.flatten()) / len(self.y_test)
        return accuracy

if __name__ == '__main__':
    X, y = datasets.load_breast_cancer(return_X_y=True)
    logistic_regression_model = LogisticRegression(X, y)
    logistic_regression_model.split_data()
    logistic_regression_model.initialize_variables()
    logistic_regression_model.train()

    accuracy = logistic_regression_model.accuracy()
    print(accuracy)
