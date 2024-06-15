from layer import Layer
from neural_net import NeuralNet
from mnist_loader import DataLoader, ImageLoader
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 20
EPOCHS = 600

myNN = NeuralNet(0.01).add_layer(Layer(3, 6)).add_layer(Layer(6, 2))

point1 = np.array([1, 2, 3]).reshape(-1, 1)
point2 = np.array([3, 2, 1]).reshape(-1, 1)
lbl1 = np.array([1, 0]).reshape(-1, 1)
lbl2 = np.array([0, 1]).reshape(-1, 1)

points = np.array([[1, 2, 3], [3, 2, 1]]).T
labels = np.array([[1, 0], [0, 1]]).T


points2 = np.array([[1, 5, 10], [5, 4, 3]]).T
labels2 = np.array([[1, 0], [0, 1]]).T


points3 = np.array([[1, 5, 10], [5, 4, 3]]).T
labels3 = np.array([[1, 0], [0, 1]]).T

points4 = np.array([[1, 3, 6], [8, 5, 1]]).T
labels4 = np.array([[1, 0], [0, 1]]).T

for i in range(EPOCHS):
    predicted = myNN.feed_forward(points)
    cost = myNN.compute_cost(labels, predicted)
    myNN.back_propagate(labels, predicted)
    myNN.gradient_descent()
    print(f"{i}; cost: {cost}, predicted: {predicted.T}, expected: {labels.T}")

    predicted = myNN.feed_forward(points2)
    cost = myNN.compute_cost(labels2, predicted)
    myNN.back_propagate(labels2, predicted)
    myNN.gradient_descent()
    print(f"{i}; cost: {cost}, predicted: {predicted.T}, expected: {labels2.T}")

    predicted = myNN.feed_forward(points3)
    cost = myNN.compute_cost(labels3, predicted)
    myNN.back_propagate(labels3, predicted)
    myNN.gradient_descent()
    print(f"{i}; cost: {cost}, predicted: {predicted.T}, expected: {labels3.T}")

    predicted = myNN.feed_forward(points4)
    cost = myNN.compute_cost(labels4, predicted)
    myNN.back_propagate(labels4, predicted)
    myNN.gradient_descent()
    print(f"{i}; cost: {cost}, predicted: {predicted.T}, expected: {labels4.T}")

    predicted = myNN.feed_forward(point1)
    cost = myNN.compute_cost(lbl1, predicted)
    myNN.back_propagate(lbl1, predicted)
    myNN.gradient_descent()
    print(f"{i}; cost: {cost}, predicted: {predicted.T}, expected: {lbl1.T}")

    predicted = myNN.feed_forward(point2)
    cost = myNN.compute_cost(lbl2, predicted)
    myNN.back_propagate(lbl2, predicted)
    myNN.gradient_descent()
    print(f"{i}; cost: {cost}, predicted: {predicted.T}, expected: {lbl2.T}")

    print(f"----------------------------------------------")


'''
sample_data = np.array([30, 20, 10, 14, 2, 40, 4, -20, 20, 10])
sample_truth = np.array([1, 0, 0])

for i in range(5000):
    predicted = myNN.feed_forward(sample_data)
    cost = myNN.compute_cost(sample_truth, predicted)
    myNN.back_propagate(sample_truth, predicted)
    myNN.gradient_descent()

    print(f"epoch {i}")
    print(f"cost: {cost}")
    print(f"predicted={predicted}, expected={sample_truth}")
    print("--------------------")
s'''
