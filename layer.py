import numpy as np
from numpy.typing import NDArray


class Layer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size  # controls k of weights matrix
        self.output_size = output_size  # controls j of weights matrix -> this is the number of neurons we have in

        self.z = np.zeros(self.output_size, np.float64)  # to be fed into activation function; z = wx + b
        self.activation = np.zeros(self.output_size, np.float64)  # activation of neuron; a = A(z)

        self.last_input = np.zeros(self.input_size, np.float64)

        self.weights = np.random.randn(output_size, input_size)  #all connections
        self.bias = np.random.randn(output_size)  # all bias

        #self.weights = np.full((output_size, input_size), 0.5)
        #self.bias = np.full(output_size, 0.5)

        self.error = np.zeros(self.output_size, np.float64)
        self.delta_bias = np.zeros(self.output_size, np.float64)
        self.delta_weights = np.zeros((self.output_size, self.input_size), np.float64)

    def load_layer(self, path):
        self.weights = np.load(path + "_weights.npy")
        self.bias = np.load(path + "_bias.npy")

    def save_layer(self, path):
        np.save(path + "_weights.npy", self.weights)
        np.save(path + "_bias.npy", self.bias)

    # feedforward step, essentially calculates and returns a (and z). input should be of size input size
    def forward(self, _input, last: bool):
        self.last_input = _input
        self.z = (self.weights @ _input) + self.bias.reshape(-1, 1)  # @ is shorthand for matrix multiplication
        if not last:
            self.activation = self.ReLu(self.z)  # could change this for different activation function
        else:
            self.activation = self.softmax(self.z)
            # print(self.activation, self.z)

        return self.activation

    # expects an array-like to operate on
    def ReLu(self, _input: NDArray):
        return np.maximum(0.1*_input, _input)

    def Relu_prime(self, _input: NDArray):
        def func(inp):
            return 1 if inp >= 0 else 0.1

        vectorised_func = np.vectorize(func)
        return vectorised_func(_input).astype(np.float64)
        # return (np.ones(_input.shape) if _input > 0 else np.full(_input.shape, 0.1)).astype(np.float64)

    def softmax(self, _input):
        _exp = np.exp(_input - np.max(_input, axis=0))
        return _exp / np.sum(_exp, axis=0)

    def softmax_prime(self, _input):
        ''' _exp = np.exp(_input - np.max(_input))
        _sum = np.sum(_exp)
        _out = (_exp * (_sum - _exp)) / _sum ** 2'''

        sm = self.softmax(_input)
        dot = np.multiply(sm, 1 - sm)
        return dot

    # expects a vector input
    def MSE_prime(self, expected: NDArray, predicted: NDArray):
        return (predicted - expected) / expected.shape[1]

    # compute error if we are the final (output) layer
    def compute_error_final(self, expected: NDArray, predicted: NDArray):
        self.error = self.MSE_prime(expected, predicted) #* self.softmax_prime(self.z)
        return self.error

    def compute_error(self, next_weights: NDArray, next_error: NDArray):
        self.error = (next_weights.T @ next_error) * self.Relu_prime(self.z)
        return self.error

    def compute_deltas(self):
        last_activation: NDArray = self.last_input

        self.delta_bias = self.error.copy()
        #self.delta_weights = np.outer(self.error, last_activation.T)
        self.delta_weights = self.error @ last_activation.T
        return self.delta_weights, self.delta_bias

    def gradient_descent(self, learning_rate):
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * np.mean(self.delta_bias, axis=1)
