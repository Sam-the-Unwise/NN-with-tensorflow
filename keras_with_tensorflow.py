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