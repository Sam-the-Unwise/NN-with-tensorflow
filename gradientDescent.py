###############################################################################
#
# AUTHOR(S): Josh Holguin
#            Samantha Muellner
#            Jacob Christiansen
# DESCRIPTION: program that will find and graph gradientDescent on the
#       provided data set -- in this case spam.data
# VERSION: 3.2.0v
#
###############################################################################

import numpy as np
import csv
from math import sqrt
import sklearn.metrics
from sklearn.preprocessing import scale



MAX_ITERATIONS = 1000
STEP_SIZE = .1



# Function: calculate_gradient
# INPUT ARGS:
#   matrix : input matrix row with obs and features
#   y_tild : modified y val to calc gradient
#   step_size : step fir gradient
#
# Return: [none]
def calculate_gradient(x_row, y_tild, step_size, weight_vector_transpose):
    # calculate elements of the denominator
    verctor_mult = np.multiply(weight_vector_transpose, x_row)
    inner_exp = np.multiply(y_tild, verctor_mult)
    denom = 1 + np.exp(inner_exp)

    numerator = np.multiply(x_row, y_tild)

    # calculate gradient
    gradient = numerator/denom

    return gradient



# Function: gradientDescent
# INPUT ARGS:
#   X : a matrix of numeric inputs {Obervations x Feature}
#   y : a vector of binary outputs {0,1}
#   stepSize : learning rate - epsilon parameters
#   max_iterations : pos int that controls how many steps to take
# Return: weight_matrix
def gradientDescent(X, y, step_size, max_iterations):
    # tuple of array dim (row, col)
    arr_dim = X.shape

    # num of input features
    X_arr_col = arr_dim[1]

    wm_total_entries = X_arr_col * max_iterations

    # variable that initiates to the weight vector
    weight_vector = np.zeros(X_arr_col)

    array_of_zeros = [0]*X_arr_col

    # for i in range(X_arr_col):
    #     array_of_zeros.append(0)

    weight_matrix = np.array(array_of_zeros)

    # ALGORITHM
    weight_vector_transpose = np.transpose(weight_vector)

    for iteration in range(0, max_iterations):

        grad_log_losss = 0

        for index in range(0, X.shape[1]):
            #calculate y_tid
            y_tild = -1

            if(y[index] == 1):
                y_tild = 1


            grad_log_losss = 0
            verctor_mult = 0
            inner_exp = 0

            # variables for simplification
            gradient = calculate_gradient(X[index,:], 
                                            y_tild, 
                                            step_size, 
                                            weight_vector_transpose)

            grad_log_losss += gradient


        mean_grad_log_loss = grad_log_losss/X.shape[1]

        # update weight_vector depending on positive or negative
        weight_vector -= np.multiply(step_size, mean_grad_log_loss)

        # store the resulting weight_vector in the corresponding 
        #   column weight_matrix
        weight_matrix = np.vstack((weight_matrix, np.array(weight_vector)))

    # get rid of initial zeros matrix that was added
    weight_matrix = np.delete(weight_matrix, 0, 0)

    weight_matrix = np.transpose(weight_matrix)

    # end of algorithm
    return weight_matrix


# # Function: scale
# # INPUT ARGS:
# #   matrix : the matrix that we need to scale
# # Return: [none]
# def scale(matrix):
#     matrix_t = np.transpose(matrix)
#     counter = 0

#     for column in matrix_t:
#         counter += 1
#         col_sq_sum = 0

#         sum = np.sum(column)
#         shape = column.shape
#         col_size = shape[0]
#         mean = sum/col_size

#         for item in column:
#             col_sq_sum += ((item - mean)**2)

#         std = sqrt(col_sq_sum/col_size)

#         column -= mean
#         column /= std



# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    data_matrix_full = np.genfromtxt( file_name, delimiter = " " )
    return data_matrix_full


# Function: split matrix
# INPUT ARGS:
#   X : matrix to be split
# Return: train, validation, test
def split_matrix(X):
    train, validation, test = np.split( X, [int(.6 * len(X)), 
                                            int(.8 * len(X))])

    return (train, validation, test)


# Function: calculate_sigmoid
# INPUT ARGS:
#   y : current vector that needs to be calculated
# Return: y_tilde_i
def calculate_sigmoid(y):
    y_tilde_i = 1/(1 + np.exp(-y))

    return y_tilde_i




# Function: main
# INPUT ARGS:
#   [none]
# Return: [none]
def main():
    # get the data from our CSV file
    data_matrix_full = convert_data_to_matrix("spam.data")
    np.random.shuffle(data_matrix_full)


    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]

    data_matrix_test = np.delete(data_matrix_full, col_length - 1, 1)


    binary_vector = data_matrix_full[:,col_length-1]
    # calculate train, test, and validation data
    #weight_matrix = calculate_train_test_and_val_data(data_matrix_test,
    #                                                data_matrix_full,
    #                                                binary_vector)

    train, validation, test = split_matrix(data_matrix_full)

    X_train_data = np.delete(train, col_length - 1, 1)
    X_validation_data = np.delete(validation, col_length - 1, 1)
    X_test_data = np.delete(test, col_length - 1, 1)

    # scale our data as needed
    scale(X_train_data)
    scale(X_validation_data)
    scale(X_test_data)

    y_train_vector = train[:,train.shape[1] - 1]
    y_validation_vector = validation[:,train.shape[1] - 1]
    y_test_vector = test[:,train.shape[1] - 1]

    train_pred_matrix = gradientDescent(X_train_data, 
                                        y_train_vector, 
                                        STEP_SIZE, 
                                        MAX_ITERATIONS)

    val_pred_matrix = gradientDescent(X_validation_data, 
                                        y_validation_vector, 
                                        STEP_SIZE, 
                                        MAX_ITERATIONS)

    test_pred_matrix = gradientDescent(X_test_data, 
                                        y_test_vector, 
                                        STEP_SIZE, 
                                        MAX_ITERATIONS)



    ###################### CALCULATE LOGISTIC REGRESSION ######################

    # get dot product of matrixes
    training_prediction = np.matmul(X_train_data, train_pred_matrix)
    validation_prediction = np.matmul(X_validation_data, val_pred_matrix)
    test_prediction = np.matmul(X_test_data, test_pred_matrix)

    sigmoid_vector = np.vectorize(calculate_sigmoid)

    # used to set numbers above 0 to 1 and below 0 to -1
    training_prediction = sigmoid_vector(training_prediction)
    validation_prediction = sigmoid_vector(validation_prediction)
    test_prediction = sigmoid_vector(test_prediction)


    # calculate minumum
    train_sum_matrix = []
    validation_sum_matrix = []

    for count in range(MAX_ITERATIONS):
        mean = np.mean(y_train_vector != training_prediction[:, count-1])

        train_sum_matrix.append(mean)

    # must use enumerate otherwise get the error 
    #       ""'numpy.float64' object is not iterable"
    val_min_index, val_min_value = min(enumerate(train_sum_matrix))


    # create loss validation matrices
    training_loss_result_matrix = []
    validation_loss_result_matrix = []

    for number in range(MAX_ITERATIONS):
        train_log_loss = sklearn.metrics.log_loss(y_train_vector, 
                                                training_prediction[:, number])
        val_log_loss = sklearn.metrics.log_loss(y_validation_vector, 
                                                validation_prediction[:, number])

        training_loss_result_matrix.append(train_log_loss)
        validation_loss_result_matrix.append(val_log_loss)


    ########################### CALCULATE ROC CURVE ###########################

    # calculate roc curves for logistic regression and baseline
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test_vector, 
                                    test_prediction[:, val_min_index])


    ######################### CALCULATE PERCENT ERROR #########################
    train_percent_error = []
    validation_percent_error = []
    test_percent_error = []

    for num in range(MAX_ITERATIONS):
        # compare prediction matrix with original vector to see if results are correct
        train_mean = np.mean(training_prediction[:, num] != y_train_vector)
        val_mean = np.mean(validation_prediction[:, num] != y_validation_vector)
        test_mean = np.mean(test_prediction[:, num] != y_test_vector)

        train_percent_error.append(train_mean)
        validation_percent_error.append(val_mean)
        test_percent_error.append(test_mean)


    
    return test_percent_error, [fpr, tpr, thresholds]



# call our main
main()
