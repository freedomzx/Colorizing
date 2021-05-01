import numpy as np
import pandas as pd
import math
import random
from PIL import Image

# images
img = Image.open('shiba.jpg')
img_width, img_height = img.size

def get_rgb_data():
    # img = Image.open('shiba.jpg')
    # img_object = img.load()
    # img_width, img_height = img.size
    # pixel_values = list(img.getdata())
    
    # uint8 contains numbers between 0 to 255, useful for colors
    rgb_data = list(img.getdata())

    # print(rgb_data)

    # create image from array
    # img_rgb = Image.fromarray(rgb_data)
    # img_rgb.save('shiba_rgb.png')
    # img_rgb.show()
    return rgb_data
    
def get_grayscale_data():
    # img = Image.open('shiba.jpg')
    # img_object = img.load()
    # img_width, img_height = img.size
    print(img_width)
    print(img_height)
    # pixel_values = list(img.getdata())
    
    # uint8 contains numbers between 0 to 255, useful for colors
    gray_data = np.zeros((img_height, img_width), dtype=np.uint8)

    for r in range(img_width):
        for c in range(img_height):
            red, green, blue = img.getpixel((r,c))
            gray_data[c][r] = (0.21*red) + (0.72*green) + (0.07*blue)

    # create image from array
    # img_gray = Image.fromarray(gray_data)
    # img_gray.save('shiba_gray.png')
    # img_gray.show()

    return gray_data

def euclidean_distance(x1, x2):
    distance = 0

    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2
    result = math.sqrt(distance)
    return result

def find_closest_center(centers, data):
    closest_center = {}
    for i in range(5):
        closest_center[i] = []
    # for each data point
    for i in range(len(data)):
        distance = []
        # append the distance between the point and all centers
        for j in range(5):
            distance.append(euclidean_distance(data[i], centers[j]))
        # get the shortest distance and cluster number (0-4) and map data point to it
        cluster_num = np.argmin(distance)
        closest_center[cluster_num].append(data[i])
    return closest_center

def replace_centers(assigned_centers, data):
    # update center location by averaging the points in each cluster
    new_centers = {}
    new_centers_r = {}
    for center_num in range(5):
        # (115.64212031,  84.07828339,  80.98021703)
        new_centers[center_num] = np.average(assigned_centers[center_num], axis = 0)
        # round tuple to whole numbers
        new_centers_r[center_num] = (int(round(new_centers[center_num][0])), int(round(new_centers[center_num][1])), int(round(new_centers[center_num][2])))
    return new_centers_r

def k_means(rgb_data):
    print("length: ", len(rgb_data))
    # get data points of centers k = 5
    centers = {}
    clusters = {}
    for i in range(5):
        centers[i] = random.choice(rgb_data)
    print("initial centers: " + str(centers))
    
    # recentering 8 times for best results
    for i in range(7):
        # assign data points to one of 5 centers based on distance (cluster)
        clusters = find_closest_center(centers, rgb_data)
        # recenter the centers by averaging the points
        centers = replace_centers(clusters, rgb_data)
        print(centers)
    return clusters, centers

def k_means_recolor(clusters, colors):
    # uint8 contains numbers between 0 to 255, useful for colors
    recolored_data = np.zeros((img_height, img_width), dtype=(np.uint8, 3))

    for r in range(img_width):
        for c in range(img_height):
            red, green, blue = img.getpixel((r,c))
            current_pixel = (red, green, blue)
            get_nearest = min(colors, key=lambda color: euclidean_distance(color, current_pixel))
            recolored_data[c][r] = get_nearest

    # create image from array
    img_rgb = Image.fromarray(recolored_data)
    img_rgb.save('shiba_recolored.png')
    img_rgb.show()
    

rgb_data = get_rgb_data()
gray_data = get_grayscale_data()
clusters, centers = k_means(rgb_data)
centers_list = []
for key, value in centers.items():
    temp = value
    centers_list.append(temp)
print(centers_list)
k_means_recolor(clusters, centers_list)
# k_means_recolor(clusters, centers)

# def grayscale(pixel_values):

