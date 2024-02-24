import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression:

    def __init__(self, X, y, learning_rate=0.001, number_of_epochs=10000):
        self.X = X
        self.y = y.reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.num_training_examples, self.num_features = None, None

        self.learning_rate = learning_rate
        self.num_epochs = number_of_epochs

        self.weights = None
        self.bias = None

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.num_training_examples, self.num_features = self.X_train.shape

    def initialize_variables(self):
        self.weights = np.random.random(self.num_features).reshape(-1, 1)
        self.bias = 0

    def forward_pass(self):
        return np.dot(self.X_train, self.weights) + self.bias

    def compute_gradients(self, Y_predictions):
        gradient_weights = (1 / self.num_training_examples) * np.dot(np.transpose(self.X_train), (Y_predictions - self.y_train))
        gradient_bias = (1 / self.num_training_examples) * np.sum(Y_predictions - self.y_train)
        return gradient_weights, gradient_bias

    def update_parameters(self, gradient_weights, gradient_bias):
        self.weights -= self.learning_rate * gradient_weights
        self.bias -= self.learning_rate * gradient_bias

    def train(self):
        for epoch in range(self.num_epochs):
            Y_predictions = self.forward_pass()
            gradient_weights, gradient_bias = self.compute_gradients(Y_predictions)
            self.update_parameters(gradient_weights, gradient_bias)

    def test(self):
        return np.dot(self.X_test, self.weights) + self.bias

    def calculate_mean_squared_error(self):
        y_predictions = self.test()
        return np.mean(np.square(self.y_test - y_predictions))

    def plot_results(self):
        y_predictions = self.test()

        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(self.X_train[:, 0], self.y_train, color=cmap(0.9), label='Training Data', s=10)
        ax.scatter(self.X_test[:, 0], self.y_test, color=cmap(0.5), label='Testing Data', s=10)
        ax.plot(self.X_test[:, 0], y_predictions, color='black', linewidth=2, label='Prediction')


        ax.set_xlabel('X-axis label')
        ax.set_ylabel('Y-axis label')
        ax.set_title('Linear Regression Results')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

    regression_model = LinearRegression(X, y)
    regression_model.split_data()
    regression_model.initialize_variables()
    regression_model.train()
    mse = regression_model.calculate_mean_squared_error()
    regression_model.plot_results()
    print(mse)

