# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sns


def part_a():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.
    # Hint: use io.imread to read in the image file

    img1 = None

    # BEGIN YOUR CODE HERE
    #useful link: https://towardsdatascience.com/how-to-use-singular-value-decomposition-svd-for-image-classification-in-python-20b1b2ac4990
    img = io.imread('image1.jpg', as_gray=True)
    u, s, v = np.linalg.svd(img, full_matrices=True)
    # END YOUR CODE HERE
    return u, s, v

def part_b(u, s, v):
    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    #compute variance
    var_explained = np.round(s**2/np.sum(s**2), decimals=3)
    sns.barplot(x=list(range(1,11)),
                y=var_explained[0:10], color='dodgerblue')
    plt.xlabel('Singular Vector', fontsize=16)
    plt.ylabel('Variance Explained', fontsize=16)
    plt.tight_layout()
    plt.savefig('img_svd')

    #Reconstruct image with top 1 singular value
    n = 1
    rank1approx = np.matrix(u[:, :n]) * np.diag(s[:n]) * np.matrix(v[:n, :])
    plt.imsave('p4_b.png', rank1approx)
    #plt.imshow(rank1approx, cmap='gray')
    #plt.show()
    # END YOUR CODE HERE
    return rank1approx

def part_c(u, s, v):
    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    n = 20
    rank2approx = np.matrix(u[:, :n]) * np.diag(s[:n]) * np.matrix(v[:n, :])
    plt.imsave('p4_c.png', rank2approx)
    #plt.imshow(rank2approx, cmap='gray')
    #plt.show()
    # END YOUR CODE HERE
    return rank20approx


if __name__ == '__main__':
    u, s, v = part_a()
    rank1approx = part_b(u, s, v)
    rank20approx = part_c(u, s, v)