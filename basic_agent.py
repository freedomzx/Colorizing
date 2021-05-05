import numpy as np
import pandas as pd
import math
import random
import threading
from collections import Counter
from PIL import Image

# images
img = Image.open('shiba_big.jpg')
img_width, img_height = img.size

k = 5

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

def get_rgb_data2():
    # uint8 contains numbers between 0 to 255, useful for colors
    rgb_data2 = np.zeros((img_height, img_width), dtype=(np.uint8, 3))

    for r in range(img_width):
        for c in range(img_height):
            red, green, blue = img.getpixel((r,c))
            rgb_data2[c][r] = (red, green, blue)

    # create image from array
    # img_rgb = Image.fromarray(rgb_data)
    # img_rgb.save('shiba_rgb.png')
    # img_rgb.show()
    return rgb_data2
    
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
        distance += (int(x1[i]) - int(x2[i]))**2
    result = math.sqrt(distance)
    return result

def find_closest_center(centers, data):
    closest_center = {}
    for i in range(k):
        closest_center[i] = []
    # for each data point
    for i in range(len(data)):
        distance = []
        # append the distance between the point and all centers
        for j in range(k):
            distance.append(euclidean_distance(data[i], centers[j]))
        # get the shortest distance and cluster number (0-4) and map data point to it
        cluster_num = np.argmin(distance)
        closest_center[cluster_num].append(data[i])
    return closest_center

def replace_centers(assigned_centers, data):
    # update center location by averaging the points in each cluster
    new_centers = {}
    new_centers_r = {}
    for center_num in range(k):
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
    for i in range(k):
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
            # goes through 5 colors, match the pixel that is most similar to one of 5 colors
            get_nearest = min(colors, key=lambda color: euclidean_distance(color, current_pixel))
            recolored_data[c][r] = get_nearest

    # create image from array
    # img_recolored = Image.fromarray(recolored_data)
    # img_recolored.save('shiba_recolored.png')
    # img_recolored.show()
    return recolored_data
    
def get_adjacent(data, row: int, col: int, width, height):
    """
    Returns adjacent neighbors' coordinates.
    :param col:
    :param row:
    :return:
    """
    neighbors = []

    # North west
    if row > 0 and col > 0:
        neighbors.append(data[col - 1][row - 1])
    # North east
    if col > 0 and row + 1 < width:
        neighbors.append(data[col - 1][row + 1])
    # South west
    if col + 1 < height and row > 0:
        neighbors.append(data[col + 1][row - 1])
    # South east
    if col + 1 < height and row + 1 < width:
        neighbors.append(data[col + 1][row + 1])
        
    # left
    if col > 0:
        neighbors.append(data[col - 1][row])
    # right
    if col + 1 < height:
        neighbors.append(data[col + 1][row])
    # down
    if row > 0:
        neighbors.append(data[col][row - 1])
    # up
    if row + 1 < width:
        neighbors.append(data[col][row + 1])
    

    # print(f"neighbors: {neighbors}")
    return neighbors

def most_frequent(rgb_values):
    # https://stackoverflow.com/questions/18827897/python-get-most-frequent-item-in-list
    lst = Counter(rgb_values).most_common()
    if not lst:
        return (150, 133, 124)
    highest_count = max([i[1] for i in lst])
    values = [i[0] for i in lst if i[1] == highest_count]
    random.shuffle(values)

    # print("lst: ", lst)
    # print("highest: ", lst[0])
    return values[0]

def get_six_patches(data, width, height, test_patch, recolored_data):
    similar_patches = []
    color_rep = []
    count = 0

    for r in range(width):
        for c in range(height):
            if r == 0 or c == 0 or r == width-1 or c == height-1:
                continue
            if count == 6:
                majority = most_frequent(color_rep)
                return similar_patches, majority
                
            curr_patch = []

            middle = data[c][r]
            # middle patch will be first
            curr_patch.append(middle)
            curr_patch.extend(get_adjacent(data, r, c, width, height))
            
            # check if two patches are similar (within a certain distance)
            distance_between = euclidean_distance(test_patch, curr_patch)
            # print("distance: ", distance_between)
            if distance_between <= 70:
                similar_patches.append(curr_patch)
                count += 1

                # get color representative from recolored
                color_rep.append(tuple(recolored_data[c][r]))

    majority = most_frequent(color_rep)
    return similar_patches, majority


def basic_agent():
    rgb_data = get_rgb_data()
    gray_data = get_grayscale_data()
    clusters, centers = k_means(rgb_data)

    # convert dict to list
    centers_list = []
    for key, value in centers.items():
        temp = value
        centers_list.append(temp)
    print("centers: ", centers_list)

    recolored_data = k_means_recolor(clusters, centers_list)

    # split img to train and test
    split_width = img_width // 2

    train_gray = gray_data[:, :split_width]
    train_recolored = recolored_data[:, :split_width]

    test_gray = gray_data[:, split_width:]
    test_recolored = recolored_data[:, split_width:]

    result = np.zeros((img_height, split_width), dtype=(np.uint8,3))

    # to_image = Image.fromarray(train_recolored)
    # print(list(train_recolored)[0][1])

    # select a color for the middle pixel of patch
    # iterate through test data (right half of gray image)
    for r in range(img_width // 2):
        for c in range(img_height):
            # ignore edges
            if r == 0 or c == 0 or r == split_width-1 or c == img_height-1:
                continue
            middle = test_gray[c][r]
            # assemble the 3x3 patch, middle pixel will always be first
            patch = []
            patch.append(middle)
            patch.extend(get_adjacent(test_gray, r, c, split_width, img_height))
            # get six most similar 3x3 grayscale pixel patches in training data (left half of gray image)
            six_patches, majority = get_six_patches(train_gray, split_width, img_height, patch, train_recolored)
            # print("six patches: ", six_patches)

            # recolor middle pixel from test gray
            result[c][r] = majority

    return train_recolored, result

def basic_agent_threaded():

    rgb_data = get_rgb_data()
    gray_data = get_grayscale_data()
    clusters, centers = k_means(rgb_data)

    # convert dict to list
    centers_list = []
    for key, value in centers.items():
        temp = value
        centers_list.append(temp)
    print("centers: ", centers_list)

    recolored_data = k_means_recolor(clusters, centers_list)

    # split img to train and test
    split_width = img_width // 2

    train_gray = gray_data[:, :split_width]
    train_recolored = recolored_data[:, :split_width]

    test_gray = gray_data[:, split_width:]
    test_recolored = recolored_data[:, split_width:]

    result = np.zeros((img_height, split_width), dtype=(np.uint8,3))
    result1 = np.zeros((math.ceil(img_height/3), split_width), dtype=(np.uint8, 3))
    result2 = np.zeros((math.ceil(img_height/3), split_width), dtype=(np.uint8, 3))
    result3 = np.zeros((math.ceil(img_height/3), split_width), dtype=(np.uint8, 3))

    thread1 = threading.Thread(target=basic_agent_thread_action, args=(result, 1, img_height, img_width, test_gray, train_gray, train_recolored))
    thread2 = threading.Thread(target=basic_agent_thread_action, args=(result, 2, img_height, img_width, test_gray, train_gray, train_recolored))
    thread3 = threading.Thread(target=basic_agent_thread_action, args=(result, 3, img_height, img_width, test_gray, train_gray, train_recolored))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()

    # result1 = np.concatenate((result1, result2), axis=None)
    # result1 = np.concatenate((result1, result3), axis=None)

    return train_recolored, result

def basic_agent_thread_action(resArray, section, length, width, testG, trainG, trainRec):
    split_width = width // 2
    first_horizontal_border = math.floor(length/3)
    second_horizontal_border = first_horizontal_border * 2
    start = 0
    end = 0
    if section == 1:
        end = first_horizontal_border-1
        print(str(start) + " " + str(end))
    elif section == 2:
        start = first_horizontal_border
        end = second_horizontal_border-1
        print(str(start) + " " + str(end))
    else:
        start = second_horizontal_border
        end = length
        print(str(start) + " " + str(end))

    # select a color for the middle pixel of patch
    # iterate through test data (right half of gray image)
    for r in range(width // 2):
        for c in range(start, img_height):
            # ignore edges
            if r == 0 or c == 0 or r == split_width-1 or c == img_height-1:
                continue
            middle = testG[c][r]
            # assemble the 3x3 patch, middle pixel will always be first
            patch = []
            patch.append(middle)
            patch.extend(get_adjacent(testG, r, c, split_width, img_height))
            # get six most similar 3x3 grayscale pixel patches in training data (left half of gray image)
            six_patches, majority = get_six_patches(trainG, split_width, img_height, patch, trainRec)
            # print("six patches: ", six_patches)

            # recolor middle pixel from test gray
            resArray[c][r] = majority



# accuracy
# score = mean_squared_error(test_recolored.tolist(), result.tolist())
# print("mean_squared_error: ", score)

train_recolored, result = basic_agent_threaded()

combine = np.concatenate((train_recolored, result), axis=1)

img_result = Image.fromarray(combine)
img_result.save('shiba_big_result.png')
img_result.show()

# k_means_recolor(clusters, centers)

# def grayscale(pixel_values):

