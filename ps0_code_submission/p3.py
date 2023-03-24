# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def part_a():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.
    # Hint: use io.imread to read in the files

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = io.imread(fname="image1.jpg")
    img2 = io.imread(fname="image2.jpg")
    # END YOUR CODE HERE
    return img1, img2

def normalize_img(img):
    pass # TODO implement this helper function for parts b and c

def part_b(img1, img2):
    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    # END YOUR CODE HERE
    return img1, img2
    
def part_c(img1, img2):
    # ===== Problem 3c =====
    # Add the images together and re-normalize them
    # to have minimum value 0 and maximum value 1.
    # Display this image.
    sumImage = None
    
    # BEGIN YOUR CODE HERE
    sumImage = img1 + img2
    sumImage = (sumImage - np.min(sumImage)) / (np.max(sumImage) - np.min(sumImage))
    plt.imsave('p3_c.png', sumImage)
    # END YOUR CODE HERE
    return sumImage

def part_d(img1, img2):
    # ===== Problem 3d =====
    # Create a new image such that the left half of
    # the image is the left half of image1 and the
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    image1 = img1[:,:150,:]
    image2 = img2[:,150:,:]
    newImage1 = np.concatenate((image1, image2), axis=1)
    plt.imsave('p3_d.png', newImage1)
    # END YOUR CODE HERE
    return newImage1

def part_e(img1, img2):    
    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd
    # numbered row is the corresponding row from image1 and the
    # every even row is the corresponding row from image2.
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    rows = img1.shape[0]
    newImage2 = np.zeros((rows, rows, 3))
    for i in range(rows):
        if i%2 == 0: #odd rows
            newImage2[i,:,:] = img1[i,:,:]
        else:
            newImage2[i,:,:] = img2[i,:,:]
    plt.imsave('p3_e.png', newImage2)
    # END YOUR CODE HERE
    return newImage2

def part_f(img1, img2):     
    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and tile may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    rows = img1.shape[0]
    #Step1: generate vector having 1's @ odd and even places
    odd = np.tile([1,0], 150)
    even = np.tile([0,1], 150)

    #Step 2: reshape to column vector
    odd = np.reshape(odd, (rows,1,1))
    even = np.reshape(even, (rows,1,1))
    newImage3 = (odd * img1) + (even * img2)
    plt.imsave('p3_f.png', newImage3)
    # END YOUR CODE HERE
    return newImage3

def part_g(img):         
    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image.
    # Display the grayscale image with a title.
    # Hint: use np.dot and the standard formula for converting RGB to grey
    # greyscale = R*0.299 + G*0.587 + B*0.114

    # BEGIN YOUR CODE HERE
    r = img[:,:,0] * 0.299
    g = img[:,:,1] * 0.587
    b = img[:,:,2] * 0.114
    img = r+g+b
    plt.imshow(img, cmap='gray')
    plt.title('Grayscale')
    plt.show()
    # END YOUR CODE HERE
    return img

if __name__ == '__main__':
    img1, img2 = part_a()
    img1, img2 = part_b(img1, img2)
    sumImage = part_c(img1, img2)
    newImage1 = part_d(img1, img2)
    newImage2 = part_e(img1, img2)
    newImage3 = part_f(img1, img2)
    img = part_g(newImage3)
