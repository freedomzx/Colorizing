import numpy as np
from PIL import Image

def grayscale():
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

    img_gray = Image.fromarray(img_gray_data)
    img_gray.save('puffin_gray.png')
    img_gray.show()

grayscale()

# def grayscale(pixel_values):

