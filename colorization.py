import numpy as np
from PIL import Image

def get_rgb_data():
    img = Image.open('puffin.jpg')
    # img_object = img.load()
    img_width, img_height = img.size
    # pixel_values = list(img.getdata())
    
    # uint8 contains numbers between 0 to 255, useful for colors
    img_rgb_data = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for r in range(img_width):
        for c in range(img_height):
            red, green, blue = img.getpixel((r,c))
            img_rgb_data[c][r] = (red, green, blue)

    # create image from array
    img_rgb = Image.fromarray(img_rgb_data)
    img_rgb.save('puffin_rgb.png')
    img_rgb.show()
    return img_rgb_data
    
def get_grayscale_data():
    img = Image.open('puffin.jpg')
    # img_object = img.load()
    img_width, img_height = img.size
    print(img_width)
    print(img_height)
    # pixel_values = list(img.getdata())
    
    # uint8 contains numbers between 0 to 255, useful for colors
    img_gray_data = np.zeros((img_height, img_width), dtype=np.uint8)

    for r in range(img_width):
        for c in range(img_height):
            red, green, blue = img.getpixel((r,c))
            img_gray_data[c][r] = (0.21*red) + (0.72*green) + (0.07*blue)

    # create image from array
    img_gray = Image.fromarray(img_gray_data)
    img_gray.save('puffin_gray.png')
    img_gray.show()

    return img_gray_data

get_rgb_data()
get_grayscale_data()

# def grayscale(pixel_values):

