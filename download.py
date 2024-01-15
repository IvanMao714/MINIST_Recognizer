import cv2
import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("./data/mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def load():
    if not os.path.exists("./data/mnist.pkl"):
        if not os.path.exists("./data"):
            os.mkdir('./data')
        download_mnist()
        save_mnist()
    with open("./data/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    print(type(mnist["training_labels"]))
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def load_mnist_data(data_path):

    images = []
    labels = []

    for label_folder in os.listdir(data_path):
        label_path = os.path.join(data_path, label_folder)


        if os.path.isdir(label_path):
            label = int(label_folder)

            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


if __name__ == '__main__':
    train_images, train_labels = load_mnist_data('./data/MINIST/train')
    test_images, test_labels,i,l = load()
