from layer import Layer
from neural_net import NeuralNet
from mnist_loader import DataLoader, ImageLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 100
EPOCHS = 30

costs = np.array([])
accuracies = np.array([])
trending_guess = np.array([])


def train_single_epoch(epoch):
    global costs, accuracies, trending_guess
    batches_with_one_guess = {}
    for i in tqdm(range(0, len(loader.train_set), BATCH_SIZE)):
        images, labels = loader.next_batch_train()
        labels_one_hot = np.zeros((10, BATCH_SIZE))
        for index, element in enumerate(labels):
            labels_one_hot[int(element), index] = 1.0

        predicted = myNN.feed_forward(images.T)
        cost = myNN.compute_cost(labels_one_hot, predicted)
        myNN.back_propagate(labels_one_hot, predicted)
        myNN.gradient_descent()

        guesses = np.argmax(predicted, axis=0)

        correct_guesses = np.equal(guesses, labels.astype(np.int64))

        accuracies = np.append(accuracies, 100 * np.count_nonzero(correct_guesses) / BATCH_SIZE)
        costs = np.append(costs, np.mean(cost))

        if np.min(guesses) == np.max(guesses):
            if guesses[0] in batches_with_one_guess:
                batches_with_one_guess[guesses[0]] += 1
            else:
                batches_with_one_guess[guesses[0]] = 1
        '''
        correct = 0
        avg_cost = 0
        for j in range(len(images)):
            image = images[j].reshape(-1, 1)
            label = np.zeros(10).reshape(-1, 1)  # one hot array time !
            label[int(labels[j]), 0] = 1.0  # this only works because index and value are equal for digits case!

            predicted = myNN.feed_forward(image)
            cost = myNN.compute_cost(label, predicted)
            myNN.back_propagate(label, predicted)
            myNN.gradient_descent()
            guesses = np.append(guesses, np.argmax(predicted))

            print(f"batch {i}, index: {j}")
            print(f"cost: {cost}")
            print(f"predicted={np.argmax(predicted)}, expected={np.argmax(label)}")
           # print(f"predicted={str(predicted)}, expected={label}")

            avg_cost += cost
            if np.argmax(predicted) == np.argmax(label):
                correct += 1
            print("--------------------")
        
        costs = np.append(costs, min(avg_cost/BATCH_SIZE, 100000))
        accuracies = np.append(accuracies, 100 * correct / BATCH_SIZE)
        '''

    print(f"all guesses were the same: {batches_with_one_guess}")
    print(f"epoch {epoch}")
    print(f"cost: {np.mean(costs)}")
    print(f"accuracy = {np.mean(accuracies)}")


# these two lines to either start new NN , or continue training saved model:

# myNN = NeuralNet(0.01).add_layer(Layer(784, 32)).add_layer(Layer(32, 32)).add_layer(Layer(32, 10))
myNN = NeuralNet(0.1).load_network("saved_network")

train_loader = ImageLoader(".\\digits_set\\train-images.idx3-ubyte", ".\\digits_set\\train-labels.idx1-ubyte",
                           60000)

test_loader = ImageLoader(".\\digits_set\\t10k-images.idx3-ubyte", ".\\digits_set\\t10k-labels.idx1-ubyte",
                          10000)

loader = DataLoader(train_loader, test_loader, BATCH_SIZE, True)
for i in range(EPOCHS):
    train_single_epoch(i)

plt.plot(trending_guess)
plt.show()
plt.plot(costs)
plt.show()
plt.plot(accuracies)
plt.show()

print("saving network to saved_network")
myNN.save_network("saved_network")
