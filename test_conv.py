import numpy as np

image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/mnist/"
train_data = np.loadtxt(data_path + "mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",
                       delimiter=",")

train_data = train_data[:800]
test_data = test_data[:200]

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr == train_labels).astype(float)
test_labels_one_hot = (lr == test_labels).astype(float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 0] = 0.01
test_labels_one_hot[test_labels_one_hot == 1] = 0.99

import conv_nn as cnn

CNN = cnn.ConvNeuralNetwork(no_of_cnn_kernels=20,
                            conv_kernel_size=9,
                            image_size=28,
                            no_of_out_nodes=10,
                            no_of_hidden_nodes=100,
                            learning_rate=0.1, )

for i in range(len(train_imgs)):
    CNN.train(train_imgs[i], train_labels_one_hot[i])
