from layer import Layer
from neural_net import NeuralNet
from mnist_loader import DataLoader, ImageLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def test():
    total_correct = 0
    for i in tqdm(range(len(loader.test_set))):
        image, label = loader.test_set[i]
        image = image.flatten()
        label_one_hot = np.zeros((10, 1))
        label_one_hot[label, 0] = 1.0

        predicted = myNN.feed_forward(image.reshape(-1, 1))

        guesses = np.argmax(predicted, axis=0)

        correct_guesses = np.equal(guesses, label)
        total_correct += np.count_nonzero(correct_guesses)

    print("test set:")
    print(f"accuracy = {100 * total_correct / len(loader.test_set)}")


myNN = NeuralNet(1).load_network("saved_network")
train_loader = ImageLoader(".\\digits_set\\train-images.idx3-ubyte", ".\\digits_set\\train-labels.idx1-ubyte",
                           60000)

test_loader = ImageLoader(".\\digits_set\\t10k-images.idx3-ubyte", ".\\digits_set\\t10k-labels.idx1-ubyte",
                          10000)

loader = DataLoader(train_loader, test_loader, 1, True)

image, label = test_loader[9230]

predicted = myNN.feed_forward(image.flatten().reshape(-1, 1))
guess = np.argmax(predicted, axis=0)

plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()

print(f"predicted: {guess}, actual: {label}")

# test full dataset
test()
