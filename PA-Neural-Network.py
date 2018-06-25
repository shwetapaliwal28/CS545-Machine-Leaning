# CS545 : Programming Assignment-1 Neural Network
# Shweta Paliwal

#A neural network with one hidden layer to recognize MNIST handwritten digits. 
#The network has 785 input units, 10 output units with varying number of hidden units (20, 50 and 100). 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

output_size = 10 
input_size = 785 # image_size = 28 * 28 , plus a bias node
train_size = 60000
test_size = 10000
no_of_iterations=50 # Run for 50 Epochs
hidden_units = [20, 50, 100]
momentum = [0, 0.25, 0.5]
eta = 0.1 # Learning rate
scale_by = 255

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y) 

# Method to draw a plot of both training and test accuracy as a function of epoch number for 50 Epochs.
def drawplot(train_accuracy,test_accuracy):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axis([0, 50, 0, 100])
    plot_epoch = range(0,51)
    test_accuracy = [i * 100 for i in test_accuracy]
    train_accuracy = [i * 100 for i in train_accuracy]
    plt.plot(plot_epoch,test_accuracy)
    plt.plot(plot_epoch,train_accuracy)   
    plt.legend(['Test Data', 'Training Data'], loc='lower right')
    plt.show()

def nntrain(wkj, wji,alpha):
    # Initialize previous change in weights to 0 :
    delp_wkj = np.zeros(wkj.shape)
    delp_wji = np.zeros(wji.shape)
    
    for i in range(0, train_size):
        # forward propogation
        img = np.reshape(train_imgs[i, :], (1, input_size))
        hj = sigmoid(np.dot(img, wji))
        hj_with_bias = np.insert(hj, 0, 1, axis=1)
        ok = sigmoid(np.dot(hj_with_bias,wkj))
        
        # backward propogation
        tk = np.insert((np.repeat(0.1, output_size - 1)), int(train_labels[i]), 0.9) #Set the target value tk for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise
        delk = dsigmoid(ok)*(tk-ok)
        delj = dsigmoid(hj_with_bias)*np.dot(delk, np.transpose(wkj))

        # Current change in weights
        delc_wkj = eta*np.dot(np.transpose(hj_with_bias), delk) + alpha*delp_wkj
        delc_wji = eta*np.dot(np.transpose(img), delj[:,1:]) + alpha*delp_wji

        #update weights
        wkj += delc_wkj
        wji += delc_wji 
    return wkj, wji

def nntest(dataset,size, wji, wkj):
    prediction = []
    for i in range(0, size):
        # forward propogation
        img = np.reshape(dataset[i, :], (1, input_size))
        hj = sigmoid(np.dot(img, wji))
        hj_with_bias = np.insert(hj, 0, 1, axis=1)
        ok = sigmoid(np.dot(hj_with_bias,wkj))
        prediction.append(np.argmax(ok)) 
    return prediction

def Neural_Network(h_size, alpha):
    # Initialize random weights between -0.05 and 0.05
    wji = np.random.uniform(low=-0.05, high=0.05, size=(input_size,h_size))
    wkj = np.random.uniform(low=-0.05, high=0.05, size=(h_size+1,output_size))
    
    test_accuracy = []
    train_accuracy = []
    epoch = 0 
    
    # Run for 50 epochs 
    while(epoch <= no_of_iterations):
        prediction = nntest(train_imgs,train_size, wji, wkj)
        train_accu = accuracy_score(train_labels, prediction)
        prediction = nntest(test_imgs,test_size, wji, wkj)
        test_accu = accuracy_score(test_labels, prediction)
        train_accuracy.append(train_accu)
        test_accuracy.append(test_accu)
        print("For Epoch " + str(epoch) + " :\n\tTraining Set Accuracy Score is " + str(train_accu) + "\n\tTest Set Accuracy Score is " + str(test_accu))
        wkj, wji = nntrain(wkj, wji,alpha)
        epoch += 1
    prediction = nntest(train_imgs,train_size, wji, wkj)
    print("For Epoch " + str(epoch) + " :\tTraining Set Accuracy Score is " + str(accuracy_score(train_labels, prediction))) 
    prediction = nntest(test_imgs,test_size,wji, wkj)
    # plotting accuracy on both the training and test data for 50 Epochs
    print("\n\nHidden Units : " + str(h_size) + "\tMomentum : " + str(alpha) + "\tTraining Size = " + str(train_size))
    drawplot(train_accuracy,test_accuracy) 
    
    #Creating a confusion matrix for trained network, summarizing results on the test set.
    print("\n\nConfusion Matrix For Test Set Accuracy Score of :"+ str(accuracy_score(test_labels, prediction))+"\n")
    print(confusion_matrix(test_labels, prediction))
    print("\n")

# Load training and test data
data_path = "/Users/shweta/Documents/MNIST/"
print("Loading train data\n")
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
print("Loading test data\n")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

# shuffle the training data for Experiment 3, so that the classes are randomly distributed.
np.random.shuffle(train_data) 

#Scale the data values to be between 0 and 1 by dividing by 255
train_data_imgs = np.asfarray(train_data[:, 1:]) / scale_by
train_imgs = np.insert(train_data_imgs, 0, 1, axis=1)
test_data_imgs = np.asfarray(test_data[:, 1:]) / scale_by
test_imgs = np.insert(test_data_imgs, 0, 1, axis=1)
train_labels = np.array(train_data[:, :1])
test_labels = np.array(test_data[:, :1])

# Experiment 1: Vary number of hidden units. Fix the momentum value to 0.9
for n in hidden_units:
    Neural_Network(n, 0.9)

# Experiment 2: Vary the momentum value.
for a in momentum:
    Neural_Network(100, a)

# Experiment 3: Vary the number of training examples. Fix the momentum value to 0.9.
#Part - 2
train_imgs, train_imgs_test, train_labels, train_labels_test = train_test_split(train_imgs, train_labels, test_size=0.75) #Change test size to 0.5 for Part -1
train_size = 15000 # train_size = 30000 for Part -1 
Neural_Network(100, 0.9)

