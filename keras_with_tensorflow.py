###############################################################################
#
# AUTHOR(S): Samantha Muellner
#            Josh Kruse
# DESCRIPTION: program that will implement a stochastic gradient descent algo
#       for a neural network with one hidden layer
# VERSION: 1.3.0v
#
###############################################################################

import numpy as np
import csv, math
from math import sqrt
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss
from sklearn.metrics import log_loss
from sklearn import neighbors, datasets
from matplotlib import pyplot as plt
import random

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.layers.core import Flatten, Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import optimizers
import tensorflow as tf

# global variables
MAX_EPOCHS = 650
STEP_SIZE = .01
BATCH_SIZE = 64

# Function: split matrix
# INPUT ARGS:
#   X_mat : matrix to be split
#   y_vec : corresponding vector to X_mat
# Return: train, validation, test
def split_matrix(X_mat, y_vec, size):
    # split data 80% train by 20% validation
    X_train, X_validation = np.split( X_mat, [int(size * len(X_mat))])
    y_train, y_validation = np.split( y_vec, [int(size * len(y_vec))])

    return (X_train, X_validation, y_train, y_validation)


# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)
    return data_matrix_full

# Function: sigmoid
# INPUT ARGS:
#   x : value to be sigmoidified
# Return: sigmoidified x
def sigmoid(x) :
    x = 1 / (1 + np.exp(-x))
    return x


# Function: main
def main():
    print("starting")
    # use spam data set
    
    data_matrix_full = convert_data_to_matrix("spam.data")
    np.random.seed( 0 )
    np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    X_Mat = np.delete(data_matrix_full, col_length - 1, 1)
    y_vec = data_matrix_full[:,57]

    X_sc = scale(X_Mat)


    # (10 points) Divide the data into 80% train, 20% test observations 
    is_train = np.random.choice( [True, False], X_sc.shape[0], p=[.8, .2] )

    # (10 points) Next divide the train data into 60% subtrain, 40% validation
    subtrain_size = np.sum( is_train == True )
    is_subtrain = np.random.choice( [True, False], subtrain_size, p=[.6, .4] )
    
    X_train = np.delete( is_train, np.argwhere(is_subtrain==True), 0)
    y_train = np.delete(is_subtrain, np.argwhere(is_subtrain == False), 0)
    X_validation = np.delete(is_train, np.argwhere(is_subtrain == False), 0)
    y_validation = np.delete(is_subtrain, np.argwhere(is_subtrain == False), 0)
    
    # (10 points) Define three different neural networks, each with one hidden layer, 
    #   but with different numbers of hidden units (10, 100, 1000). 
    #   In keras each is a sequential model with one dense layer.
    
    # define our optimizer function -- stochastic gradient descent in this case
    # All parameter gradients will be clipped to
    # a maximum norm of 1.
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

    # define/create models

    ##### MODEL 10 #####
    model_1 = Sequential()
    model_1.add(Dense(10, activation='relu'))

    # train our models
    # result_1 = model_1.fit(x = X_train, 
    #                         y = y_train, 
    #                         #batch_size = ,
    #                         epochs = MAX_EPOCHS, 
    #                         steps_per_epoch = STEP_SIZE,
    #                         validation_data = (X_validation, y_validation))

    result_1 = model_1.fit_generator(X_train,
                                steps_per_epoch = STEP_SIZE, #train_len//TRAIN_BATCH_SIZE
                                #mess around with epochs to find a better accuracy
                                epochs = MAX_EPOCHS,
                                validation_data = (X_validation, y_validation),
                                validation_steps = BATCH_SIZE) 

    result_1.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])


    # ##### MODEL 100 #####

    # model_2 = Sequential()
    # model_2.add(Dense(100, activation='relu'))
    
    
    # # result_2 = model_2.fit(x = X_train, 
    # #                         y = y_train, 
    # #                         #batch_size = ,
    # #                         epochs = MAX_EPOCHS, 
    # #                         steps_per_epoch = STEP_SIZE,
    # #                         validation_data = (X_validation, y_validation))

    
    # result_2 = model_2.fit_generator(X_train,
    #                             steps_per_epoch = STEP_SIZE, #train_len//TRAIN_BATCH_SIZE
    #                             #mess around with epochs to find a better accuracy
    #                             epochs = MAX_EPOCHS,
    #                             validation_data = (X_validation, y_validation),
    #                             validation_steps = BATCH_SIZE) 
    
    # result_2.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    # ##### MODEL 1000 #####
    # model_3 = Sequential()
    # model_3.add(Dense(1000, activation='relu'))

    # # result_3 = model_3.fit(x = X_train, 
    # #                         y = y_train, 
    # #                         #batch_size = ,
    # #                         epochs = MAX_EPOCHS, 
    # #                         steps_per_epoch = STEP_SIZE,
    # #                         validation_data = (X_validation, y_validation))

    # result_3 = model_3.fit_generator(X_train,
    #                             steps_per_epoch = STEP_SIZE, #train_len//TRAIN_BATCH_SIZE
    #                             #mess around with epochs to find a better accuracy
    #                             epochs = MAX_EPOCHS,
    #                             validation_data = (X_validation, y_validation),
    #                             validation_steps = BATCH_SIZE) 

    # result_3.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
    

    

    

    # (20 points) On the same plot, show the logistic loss as a function of 
    #   the number of epochs (use a different color for each number of 
    #   hidden units, e.g. light blue=10, dark blue=100, black=1000, 
    #   and use a different linetype for each set, 
    #   e.g. subtrain=solid, validation=dashed). 
    #   Draw a point to emphasize the minimum of each validation loss curve.


    # (10 points) For each of the three networks, define a variable called 
    #   best_epochs which is the number of epochs which minimizes the 
    #   validation loss.
    best_epoch_1 = 0
    best_epoch_2 = 0
    best_epoch_3 = 0

    # (10 points) Re-train each network on the entire train set (not just 
    #   the subtrain set), using the corresponding value of best_epochs 
    #   (which should be different for each network).
    # (10 points) Finally use the learned models to make predictions on the test set. What is the prediction accuracy? (percent correctly predicted labels in the test set) What is the prediction accuracy of the baseline model which predicts the most frequent class in the train labels?



    
    
    
    # EXTRA CREDIT
    # 10 points if your github repo includes a README.org (or README.md etc) file with a link to the source code of your script, and an explanation about how to install the necessary libraries, and run it on the data set.
    # 10 points if you do 4-fold cross-validation instead of the single train/test split described above, and you make a plot of test accuracy for all models for each split/fold.
    # 10 points if you show GradientDescent (from project 1, logistic regression with number of iterations selected by a held-out validation set) in your test accuracy result figure.
    # 10 points if you show NearestNeighborsCV (from project 2) in your test accuracy figure.
    # 10 points if you show NNOneSplit (from project 3) in your test accuracy figure.
    # 10 points if you compute and plot ROC curves for each (test fold, algorithm) combination. Make sure each algorithm is drawn in a different color, and there is a legend that the reader can use to read the figure. 