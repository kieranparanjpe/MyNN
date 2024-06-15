from abc import ABC
import random
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
import math


class ImageLoader(Sequence, ABC):
    def __init__(self, images, labels, length):
        self.images = images
        self.labels = labels
        self.length = length
        self.image_length = 28
        return

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item >= len(self):
            return np.zeros((28, 28), np.float64), -1

        lbl_f = open(self.labels, "rb")  # labels (digits)
        img_f = open(self.images, "rb")  # pixel values

        img_f.seek(16 + item * (self.image_length ** 2))
        lbl_f.seek(8 + item)

        element = np.zeros((28, 28), np.float64)

        lbl = ord(lbl_f.read(1))  # Unicode, one byte
        for j in range(self.image_length ** 2):  # get 784 pixel vals
            val = ord(img_f.read(1))
            element[math.floor(j / self.image_length), math.floor(j % self.image_length)] = (val / 255.0)

        img_f.close()
        lbl_f.close()

        return element, lbl


class DataLoader:

    def __init__(self, train_set: ImageLoader, test_set: ImageLoader, batch_size, scramble):
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.train_index = 0
        self.train_order = list(range(0, len(train_set)))
        if scramble:
            random.shuffle(self.train_order)

    def next_batch_train(self):
        batch_image = np.zeros((self.batch_size, self.train_set.image_length * self.train_set.image_length), np.float64)
        batch_label = np.zeros(self.batch_size, np.float64)

        if self.train_index >= len(self.train_order):
            self.train_index = 0
            random.shuffle(self.train_order)

        for i in range(self.train_index, self.train_index + self.batch_size):
            if i >= len(self.train_order):
                batch_image.resize(i - self.train_index, self.train_set.image_length * self.train_set.image_length)
                batch_label.resize(i - self.train_index)
                break

            index = self.train_order[i]
            img, lbl = self.train_set[index]
            batch_image[i - self.train_index] = img.flatten()
            batch_label[i - self.train_index] = lbl

        self.train_index += self.batch_size

        return batch_image, batch_label


if __name__ == "__main__":
    train_loader = ImageLoader(".\\digits_set\\train-images.idx3-ubyte", ".\\digits_set\\train-labels.idx1-ubyte",
                               60000)

    test_loader = ImageLoader(".\\digits_set\\t10k-images.idx3-ubyte", ".\\digits_set\\t10k-labels.idx1-ubyte",
                              10000)

    loader = DataLoader(train_loader, test_loader, 5)

    images, labels = loader.next_batch_train()

    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        print(f"label={label}")
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))

        plt.show()
