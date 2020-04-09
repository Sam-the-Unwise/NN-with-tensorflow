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
from matplotlib import pyplot as plt
import random

from sklearn.preprocessing import scale
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.layers.core import Flatten, Dropout, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import optimizers
import tensorflow as tf


from SG_with_early_stopping_regularization import SG_main
from nearest_neightbors import NN_main
from gradientDescent import GD_main

# global variables
MAX_EPOCHS = 1000
DATA_FILE = "spam.data"


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


# function that will create our NN model given the amount of units passed in
def create_model(units) :
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    
    model = Sequential()
    
    model.add(Dense(units=units, activation='sigmoid', use_bias=False))
    model.add(Dense(1, activation="sigmoid", use_bias=False))
    
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model


# function to plot our loss
def plot_loss( res_1, res_2, res_3 ) :
    plt.plot(res_1.history['loss'], color='#30baff', label="10 train")
    min_index = np.argmin(res_1.history['loss'])
    plt.plot(min_index, res_1.history['loss'][min_index], "go")

    plt.plot(res_1.history['val_loss'], '--', color='#30baff', label="10 val")
    res_1_best = np.argmin(res_1.history['val_loss'])
    plt.plot(res_1_best, res_1.history['val_loss'][res_1_best], "go")

    plt.plot(res_2.history['loss'], color='#185d80', label="100 train")
    min_index = np.argmin(res_2.history['loss'])
    plt.plot(min_index, res_2.history['loss'][min_index], "go")

    plt.plot(res_2.history['val_loss'], '--', color='#185d80', label="100 val")
    res_2_best = np.argmin(res_2.history['val_loss'])
    plt.plot(res_2_best, res_2.history['val_loss'][res_2_best], "go")

    plt.plot(res_3.history['loss'], color='#040f14', label="1000 train")
    min_index = np.argmin(res_3.history['loss'])
    plt.plot(min_index, res_3.history['loss'][min_index], "go")

    plt.plot(res_3.history['val_loss'], '--', color='#040f14', label="1000 val")
    res_3_best = np.argmin(res_3.history['val_loss'])
    plt.plot(res_3_best, res_3.history['val_loss'][res_3_best], "go")

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    return res_1_best+1, res_2_best+1, res_3_best+1


# Function: main
def main():
    print("starting")
    # use spam data set

    data_matrix_full = convert_data_to_matrix(DATA_FILE)
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

    X_train = np.delete( X_sc, np.argwhere( is_subtrain != True ), 0)
    y_train = np.delete( y_vec, np.argwhere( is_subtrain != True ), 0)
    X_validation = np.delete( X_sc, np.argwhere( is_subtrain != False ), 0)
    y_validation = np.delete( y_vec, np.argwhere( is_subtrain != False ), 0)
    X_test = np.delete( X_sc, np.argwhere( is_train != False ), 0 )
    y_test = np.delete( y_vec, np.argwhere( is_train != False ), 0 )

    # (10 points) Define three different neural networks, each with one hidden layer,
    #   but with different numbers of hidden units (10, 100, 1000).
    #   In keras each is a sequential model with one dense layer.

    # define our optimizer function -- stochastic gradient descent in this case
    # All parameter gradients will be clipped to
    # a maximum norm of 1.
    #sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

    print("Creating model...")
    # define/create models
    model_1 = create_model(10)
    model_2 = create_model(100)
    model_3 = create_model(1000)

    # train our models
    result_1 = model_1.fit( x = X_train,
                            y = y_train,
                            epochs = MAX_EPOCHS,
                            validation_data=(X_validation, y_validation),
                            verbose=2)

    result_2 = model_2.fit( x = X_train,
                            y = y_train,
                            epochs = MAX_EPOCHS,
                            validation_data=(X_validation, y_validation),
                            verbose=2)

    result_3 = model_3.fit( x = X_train,
                            y = y_train,
                            epochs = MAX_EPOCHS,
                            validation_data=(X_validation, y_validation),
                            verbose=2)

    # (20 points) On the same plot, show the logistic loss as a function of
    #   the number of epochs (use a different color for each number of
    #   hidden units, e.g. light blue=10, dark blue=100, black=1000,
    #   and use a different linetype for each set,
    #   e.g. subtrain=solid, validation=dashed).
    #   Draw a point to emphasize the minimum of each validation loss curve.
    # summarize history for loss
    best_epoch_1, best_epoch_2, best_epoch_3 = plot_loss(result_1, result_2, result_3)

    print("Finalizing model")
    # (10 points) Re-train each network on the entire train set (not just
    #   the subtrain set), using the corresponding value of best_epochs
    #   (which should be different for each network).
    final_model_1 = create_model(10)
    final_model_2 = create_model(100)
    final_model_3 = create_model(1000)

    final_result_1 = final_model_1.fit( x = X_sc,
                            y = y_vec,
                            epochs = best_epoch_1,
                            verbose=2)

    final_result_2 = final_model_2.fit( x = X_sc,
                            y = y_vec,
                            epochs = best_epoch_1,
                            verbose=2)

    final_result_3 = final_model_3.fit( x = X_sc,
                            y = y_vec,
                            epochs = best_epoch_1,
                            verbose=2)

    # (10 points) Finally use the learned models to make predictions on the test set. What is the prediction accuracy? (percent correctly predicted labels in the test set) What is the prediction accuracy of the baseline model which predicts the most frequent class in the train labels?
    print("Prediction accuracy (correctly labeled) for 10   hidden units :", final_model_1.evaluate(X_test,y_test)[1])
    print("Prediction accuracy (correctly labeled) for 100  hidden units :", final_model_2.evaluate(X_test,y_test)[1])
    print("Prediction accuracy (correctly labeled) for 1000 hidden units :", final_model_3.evaluate(X_test,y_test)[1])

    baseline = np.zeros(y_test.shape)
    print("Baseline prediction accuracy :", np.mean(baseline == y_test))

    

    # EXTRA CREDIT
    # 10 points if you do 4-fold cross-validation instead of the single train/test 
    #   split described above, and you make a plot of test accuracy for all models 
    #   for each split/fold.
    

    # 10 points if you show GradientDescent (from project 1, logistic regression 
    #   with number of iterations selected by a held-out validation set) in your 
    #   test accuracy result figure.
    # 10 points if you show NearestNeighborsCV (from project 2) in your test 
    #   accuracy figure
    # 10 points if you show NNOneSplit (from project 3) in your test accuracy 
    #   figure

    gd_accuracy, gd_roc_curve_info = GD_main()

    nn_list_of_elements, nn_mean_accuracy = NN_main()

    sg_v_mat, sg_w_vec, sg_loss_values = SG_main()




    # 10 points if you compute and plot ROC curves for each (test fold, 
    #   algorithm) combination. Make sure each algorithm is drawn in a different color, and there is a legend that the reader can use to read the figure.
main()


# %%
