import csv
import os
import shutil

import numpy as np
from numpy.typing import NDArray

from layer import Layer


class NeuralNet:
    def __init__(self, learning_rate):
        self.layers = np.array([])
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.layers = np.append(self.layers, layer)
        return self

    def feed_forward(self, _input: NDArray):
        x = _input
        for layer in self.layers:
            if x.shape[0] != layer.input_size:
                raise ValueError(f"input size {x.shape[0]} does not match expected input size {layer.input_size}")

            x = layer.forward(x, layer == self.layers[len(self.layers) - 1])

        return x

    # currently using MSE: c = 1/2m sum (e-p)^2
    def compute_cost(self, expected: NDArray, predicted: NDArray):
        if predicted.shape != expected.shape:
            raise ValueError(f"predicted shape of {predicted.shape} != expected shape of {expected.shape}")

        # print((expected - predicted) ** 2)
        cost = np.mean((expected - predicted) ** 2, axis=0) / 2
        return cost

    def back_propagate(self, expected, predicted):
        if len(self.layers) <= 0:
            raise IndexError("not enough layers")

        current_layer: Layer = self.layers[len(self.layers) - 1]
        next_error = current_layer.compute_error_final(expected, predicted)
        current_layer.compute_deltas()

        for i in range(len(self.layers) - 2, -1, -1):
            next_layer = current_layer
            current_layer = self.layers[i]
            next_error = current_layer.compute_error(next_layer.weights, next_error)
            current_layer.compute_deltas()

    def gradient_descent(self):
        for layer in self.layers:
            layer.gradient_descent(self.learning_rate)

    def save_network(self, folder_path):
        try:
            shutil.rmtree(folder_path)
        except OSError as e:
            print(f"Error: {e.strerror}")

        os.mkdir(folder_path)

        with open(os.path.join(folder_path, "network_details.csv"), 'w', newline='') as file:
            writer = csv.writer(file)

            for (index, layer) in enumerate(self.layers):
                writer.writerow([layer.input_size, layer.output_size])
                layer.save_layer(os.path.join(folder_path, str(index)))

    def load_network(self, folder_path):
        with open(os.path.join(folder_path, "network_details.csv"), 'r', newline='') as file:
            reader = csv.reader(file)

            for (index, row) in enumerate(reader):
                new_layer = Layer(int(row[0]), int(row[1]))
                new_layer.load_layer(os.path.join(folder_path, str(index)))
                self.add_layer(new_layer)

        return self

