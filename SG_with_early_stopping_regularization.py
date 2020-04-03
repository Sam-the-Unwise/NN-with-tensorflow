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

# global variables
MAX_EPOCHS = 650
STEP_SIZE = .01
N_HIDDEN_UNITS = 10



# Function: NNOneSplit
# INPUT ARGS:
#       X_mat : feature matrix (n_observations x n_features)
#       y_vec : label vector (n_observations x 1)
#       max_epochs : scalar int > 1
#       step_size
#       n_hidden_units : number of hidden units
#       is_train : logical vector of size n_observations
#           - TRUE if the observation is the train set
#           - FALSE for the validation set
# Return: list/dictionary containing at least three elements named:
#           - v_mat : best weight matrix (n_features x n_hidden_units) used to
#                          predict hidden units given inputs
#           - w_vec : best weight vector (n_hidden_units) used to predict output
#                          given hidden units
#           - loss_values : a matrix/data_table/etc which stores the logistic
#                          loss with respect to the train/validation set for each
#                          iteration
def NNOneSplit(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_train):
    n_features = X_mat.shape[1]
    #print(n_features)

    # initialize v_mat and w_vec to some random number close to zero
    np.random.seed( 0 )
    v_mat = np.random.randn( n_features, n_hidden_units )
    w_vec = np.random.randn( n_hidden_units )
    #print(v_mat)
    #print(w_vec)

    #print(v_mat.shape[1])

    loss_train = []
    loss_val = []

    #   for is_train, randomly assign 60% train and 40% validation
    #X_train, X_validation, y_train, y_validation = split_matrix(X_mat, y_vec, .6)
    X_train = np.delete( X_mat, np.argwhere(is_train==True), 0)
    y_train = np.delete(y_vec, np.argwhere(is_train == True), 0)
    X_validation = np.delete(X_mat, np.argwhere(is_train == False), 0)
    y_validation = np.delete(y_vec, np.argwhere(is_train == False), 0)

    #is_train = np.array([])

    X_train_i = X_train[0]

    index=0
    #print( sigmoid( v_mat[ index ] * X_train[ index ] ) )

    #v_mat = v_mat - step_size * gradient

    # during each iteration compute the gradients of v_mat and w_vec
    #       by taking a step (scaled by step_size) in the neg. gradient direction-
    for epoch in range(max_epochs):
        if( epoch%10 == 0 ) : print("Iteration ", epoch, "of ", max_epochs)
        for index in range( (X_train.shape[0]) -1 ) :
            if y_train[ index ] == 0 : y_tild = -1
            else : y_tild = 1

            h_v = sigmoid( np.transpose( v_mat ) @ X_train[ index ] )

            first_term = (1 / (1 + np.exp(-y_tild * (h_v))))
            second_term = (np.exp(-y_tild * (h_v)))
            third_term = (-y_tild * (np.transpose(v_mat * (h_v * (1 - h_v))) @ X_train[index]))
            gradient = (first_term * second_term * third_term)
            v_mat = v_mat - step_size * gradient

            #print(v_mat)
            y_hat = sigmoid( w_vec @ h_v ) # TODO: this shouldn't be this way :0
            #print(y_hat)

            first_term = (1 / (1 + np.exp(-y_tild * (y_hat))))
            second_term = (np.exp(-y_tild * (y_hat)))
            third_term = (-y_tild * (np.transpose(w_vec * (y_hat * (1 - y_hat))) @ h_v))
            gradient = (first_term * second_term * third_term)
            w_vec = w_vec - step_size * gradient

        # at each iteration compute the log. loss on the train/validation sets
        #       store in loss_values
        y_train_pred = np.around(sigmoid(sigmoid(X_train @ v_mat) @ w_vec ))
        #print(y_train_pred)
        #print("hi", np.mean( y_train != y_train_pred ))
        #loss_train.append( np.mean( y_train != y_train_pred ) )
        loss_train.append( log_loss( y_train, y_train_pred ))

        y_val_pred = np.around(sigmoid(sigmoid(X_validation @ v_mat) @ w_vec))
        #print(y_val_pred)
        #print("bye", np.mean( y_validation != y_val_pred))
        #loss_val.append( np.mean( y_validation != y_val_pred) )
        loss_val.append( log_loss( y_validation, y_val_pred ))

    loss_values = [[],[]]
    loss_values[0].append(loss_train)
    loss_values[1].append(loss_val)
    #print(loss_values)

    return v_mat, w_vec, loss_values

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
def SG_main():
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

    # First scale the input matrix (each column should have mean 0 and variance 1).
    # You can do this by subtracting away the mean and then dividing by the standard deviation of each column
    # (or just use a standard function like scale in R).
    X_sc = scale(X_Mat)

    # (5 points) Next create a variable is.train (logical vector with size equal to the number of observations
    # in the whole data set). Each element is TRUE if the corresponding observation (row of input matrix)
    # is in the train set, and FALSE otherwise.
    # There should be 80% train, 20% test observations (out of all observations in the whole data set).
    is_train = np.random.choice( [True, False], X_sc.shape[0], p=[.8, .2] )

    # (5 points) Next create a variable is.subtrain (logical vector with size equal to the
    # number of observations in the train set).
    # Each element is TRUE if the corresponding observation is is the subtrain set, and FALSE otherwise.
    # There should be 60% subtrain, 40% validation observations (out of 100% train observations).
    subtrain_size = np.sum( is_train==True )
    is_subtrain = np.random.choice( [True, False], subtrain_size, p=[.6, .4] )

    # (5 points) Use NNOneSplit with the train set as X.mat/y.vec,
    # with is.subtrain as specified above, and a large number for max.epochs.
    v_mat, w_vec, loss_values = NNOneSplit( X_sc, y_vec, MAX_EPOCHS, STEP_SIZE, N_HIDDEN_UNITS, is_subtrain )

    # (5 points) Plot the subtrain/validation loss as a function of the number of epochs,
    # and draw a point to emphasize the minimum of the validation loss curve.
    # print(loss_values[0])
    # plt.plot(loss_values[0][0], "-g", label="Train")
    # min_index = np.argmin(loss_values[0][0])
    # plt.plot(min_index, loss_values[0][0][min_index], "go")

    # plt.plot(loss_values[1][0], "-r", label="Validation")
    # min_index = np.argmin(loss_values[1][0])
    # plt.plot(min_index, loss_values[1][0][min_index], "ro")
    # plt.xlabel('epoch')
    # plt.ylabel('error rate') 

    return v_mat, w_vec, loss_values
