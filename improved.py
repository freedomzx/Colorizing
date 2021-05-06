import numpy as np
import pandas as pd
import math
import random
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('shiba.jpg')
img_width, img_height = img.size

def get_rgb_data():
    # uint8 contains numbers between 0 to 255, useful for colors
    red_data = np.zeros((img_height, img_width), dtype=np.uint8)
    green_data = np.zeros((img_height, img_width), dtype=np.uint8)
    blue_data = np.zeros((img_height, img_width), dtype=np.uint8)

    for r in range(img_width):
        for c in range(img_height):
            red, green, blue = img.getpixel((r,c))
            red_data[c][r] = red
            green_data[c][r] = green
            blue_data[c][r] = blue

    return red_data, green_data, blue_data

def get_grayscale_data():
    print(img_width)
    print(img_height)
    # pixel_values = list(img.getdata())
    
    # uint8 contains numbers between 0 to 255, useful for colors
    gray_data = np.zeros((img_height, img_width), dtype=np.uint8)

    for r in range(img_width):
        for c in range(img_height):
            red, green, blue = img.getpixel((r,c))
            gray_data[c][r] = (0.21*red) + (0.72*green) + (0.07*blue)

    return gray_data

# 1. Activation function that transform linear inputs to nonlinear outputs.
# 2. Bound output to between 0 and 1 so that it can be interpreted as a probability.
# 3. Make computation easier than arbitrary activation functions.
def sigmoid(z):
    s = 1.0/(1.0 + np.exp(-z))
    return s

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    diffAY = A - Y
    dw = np.dot(X, diffAY.T) / m
    db = np.sum(diffAY) / m
    return dw, db, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = [None] * num_iterations
    for i in range(num_iterations):
        dw, db, cost = propagate(w, b, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        costs[i] = cost
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return w, b, costs

def predict(w, b, X):
    return np.where(sigmoid(np.dot(w.T, X) + b) > 0.5, 1.0, 0.0)

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    w, b, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    return Y_prediction_test, costs

def initialize_with_zeros(dim):
    return  np.zeros([dim, 1]), 0.0

# flatten
# RGB ----------------------------------------------------------------
red_data_m, green_data_m, blue_data_m = get_rgb_data()

# GRAY ----------------------------------------------------------------
gray_data_matrix = get_grayscale_data()

# rescale values so that its between 0 and 1 instead of 0 and 255, width 80 height 76
rescale_gray = gray_data_matrix / 255.0
rescale_red = red_data_m / 255.0
rescale_green = green_data_m / 255.0
rescale_blue = blue_data_m / 255.0

# split img to train and test
split_width = img_width // 2

# independent variables------------------------------------------------------------ shape = (76, 40)
# X_train, will be used to train the model (gray image)
train_gray_m = rescale_gray[:, :split_width]
# X_test used to make predictions
test_gray_m = rescale_gray[:, split_width:]

# dependent variables-------------------------------------------------------------- shape = (76, 40, 3)
# y_train colors that need to be predicted by the model
train_red_m = rescale_red[:, :split_width]
train_green_m = rescale_green[:, :split_width]
train_blue_m = rescale_blue[:, :split_width]

# y_test used to test the accuracy between actual and predicted
# test_red_m = rescale_red[:, split_width:]

# flattened ------------------------------------------------------------------------
train_gray = np.reshape(train_gray_m, -1)
test_gray = np.reshape(test_gray_m, -1)

train_red = np.reshape(train_red_m, -1)
train_green = np.reshape(train_green_m, -1)
train_blue = np.reshape(train_blue_m, -1)