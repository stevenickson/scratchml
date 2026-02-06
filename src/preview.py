from PIL import Image
import numpy

# A row of data from our dataset consists of an array whose
# first element is the label (either 0, 1, 2, ..., or 9), and
# the next 784 elements are grayscale pixel values - and example
# might look like
#
# [5, 0, 0, 255, ...]
#
# This function, given such and array, extracts a two-dimensional array of RGB pixels
# suitable for rendering to an image. The first element of the input is ignored, and
# the remainder is reshaped into a 28 by 28 2-dimensional array.
# 
# Takes a 1-dimensional array of values from 0 to 255 of
# length 784, and displays it as a 28px by 28 px grayscale
# image
def pixel_grid_of_row(row):
    return numpy.reshape(row[1:], (28, 28))

# Given a pixel grid, as descriped by pixel_grid_of_row, this function displays the
# pixel grid as an image using the operating system's defualt image viewer
def display_pixel_grid(pixel_array):
    i = Image.fromarray(pixel_array)
    i.show()