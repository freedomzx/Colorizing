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
def sigmoid(z, w):
    temp = np.dot(w, z)
    s = 1.0/(1.0 + np.exp(-temp))
    return s

def cost(w, x, y):
    m = x.shape[1]
    # dot product of weights and X
    fx = sigmoid(x, w)
    result = - np.sum(y * np.log(fx) + (1 - y) * np.log(1 - fx)) / m
    return result

def gradient_descent(x, y, w, learning_rate, epochs):
    m = x.shape[1]
    j = [cost(w, x, y)]
    for i in range(0, epochs):
        fx = sigmoid(x, w)
        for i in range(0, x.shape[1]):
            w[i] -= (learning_rate/m) * np.sum((fx - y) * x[i])
        j.append(cost(w, x, y))
    return j, w

# def predict(x, y, weights, learning_rate, epochs):
#     j, w = gradient_descent(x, y, weights, learning_rate, epochs)
#     predictions = np.zeros((1, x.shape[1]))
#     fx = sigmoid(x, weights)

#     print(fx.shape)

#     for i in range(fx.shape[0]):
#         predictions[i] = fx[i]

#     print(len(fx))
#     print(len(y))
#     # accuracy = np.sum([y[i] == fx[i] for i in range(len(y))])/len(y)
#     return j

def initialize_with_zeros(dim):
    w = [0.5]*dim
    return w

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

# dependent variables-------------------------------------------------------------- shape = (76, 40)
# y_train colors that need to be predicted by the model
train_red_m = rescale_red[:, :split_width]
train_green_m = rescale_green[:, :split_width]
train_blue_m = rescale_blue[:, :split_width]

# y_test used to test the accuracy between actual and predicted
# test_red_m = rescale_red[:, split_width:]

# flattened ------------------------------------------------------------------------
train_gray = np.array(train_gray_m)
test_gray = np.array(test_gray_m)

train_red = np.array(train_red_m)
train_green = np.array(train_green_m)
train_blue = np.array(train_blue_m)
print("gray shape ", train_gray.shape[1])
print("red shape ", train_red.shape[1])
weights = initialize_with_zeros(train_gray.shape[0])
pp, w1 = gradient_descent(train_gray, train_red, weights, 0.005, 2000)
pp, w2 = gradient_descent(train_gray, train_green, weights, 0.005, 2000)
pp, w3 = gradient_descent(train_gray, train_blue, weights, 0.005, 2000)

results_red = []
results_green = []
results_blue = []

for i in range(test_gray.shape[0]):
    results_red.append(sigmoid(test_gray, w1))
    results_green.append(sigmoid(test_gray, w2))
    results_blue.append(sigmoid(test_gray, w3))

# results_red = np.array(results_red * 255.0/results_red.max())
# results_green = np.array(results_green * 255.0/results_green.max())
# results_blue = np.array(results_blue * 255.0/results_blue.max())

# print(results_red)
