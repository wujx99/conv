import numpy as np
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "../data/mnist/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 

train_data = train_data[:8000]
test_data = test_data[:2000]

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])



lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(float)
test_labels_one_hot = (lr==test_labels).astype(float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

import cnn_batch as cnn

CNN = cnn.ConvNeuralNetwork(no_of_cnn_kernels=20, 
                            conv_kernel_size=9,
                            image_size=28,
                    no_of_out_nodes=10, 
                    no_of_hidden_nodes=100,
                    learning_rate = 0.2,)
    
epochs = 20
print("学习率为",CNN.learning_rate)
batch_size =  80
no_of_batch = len(train_imgs) // batch_size
print("no_of_batch",no_of_batch)
for epoch in range(epochs):
    for i in range(batch_size):
        CNN.train(train_imgs[i*no_of_batch:(i+1)*no_of_batch], train_labels_one_hot[i*no_of_batch:(i+1)*no_of_batch])
        print("epoch: ",epoch+1,"batch :",i+1 )
        corrects, wrongs = CNN.evaluate(train_imgs, train_labels)
        print("accuracy train: ", corrects / ( corrects + wrongs))
        corrects, wrongs = CNN.evaluate(test_imgs, test_labels)
        print("accuracy: test", corrects / ( corrects + wrongs))
