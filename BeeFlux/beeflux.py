# -*- coding: utf-8 -*-
"""
Written By Chris Borke
Date: 10/27/2016

BreadCrumbs:
    
http://www.scipy-lectures.org/advanced/image_processing/
http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
http://www.cse.psu.edu/~rtc12/CSE486/lecture11_6pp.pdf


"""

from __future__ import division
from scipy import ndimage, misc
from skimage import feature, data, exposure
from PIL import Image
import cv2
import numpy as np

from matplotlib import pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt, pi
from skimage.color import rgb2gray

BLOB_AREA_THRESH = 1000

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
 
    def get_lbp(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        return lbp

    def get_hist(self, lbp):
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
          
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        # return the histogram of Local Binary Patterns
        return hist

def move_ball(img):
    #For the messi.jpg example
    offset = 0
    ball = img[280:(340+offset), 330:390]
    img[273:(333+offset), 100:160] = ball

def lbp_example(imagename):

    img = cv2.imread(imagename)
    lbinpat = LocalBinaryPatterns(24,8)    

    cv2.imshow('original', img)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayed', gimg)
    lbp = lbinpat.get_lbp(gimg)
    inv = invert_image(lbp)
    cv2.imshow('lbp', lbp)
    cv2.imshow('lbp', inv)
    
def blob_example(imgname):
    image = misc.imread(imgname)
    
    image_gray = rgb2gray(image)
    
    #Laplacian of Gaussian (Blue)
    fig,ax = plt.subplots(1, 1, sharex=True, sharey=True, subplot_kw={'adjustable':'box'})
    ax.imshow(image, interpolation='nearest')

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    for blob_coord in blobs_log:
        y, x, r = blob_coord
        BLOB_AREA_THRESH = (0, 50) #Alter this for some basic analysis filtering by area
       
        if BLOB_AREA_THRESH[0] < pi*r*r and pi*r*r < BLOB_AREA_THRESH[1]:
            c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
            ax.add_patch(c)
    
    plt.savefig('blobs_LoG.jpg', dpi=96*10)
    
    
    #Difference of Gaussian (Yellow)
    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    
    fig,ax = plt.subplots(1, 1, sharex=True, sharey=True, subplot_kw={'adjustable':'box'})
    ax.imshow(image, interpolation='nearest')
    
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    for blob_coord in blobs_dog:
        y, x, r = blob_coord
        BLOB_AREA_THRESH = (0, 5000) #Alter this for some basic analysis filtering by area
       
        if BLOB_AREA_THRESH[0] < pi*r*r and pi*r*r < BLOB_AREA_THRESH[1]:
            c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
            ax.add_patch(c)
    
    plt.savefig('blobs_DoG.jpg', dpi=96*10)
    
    
    
    #Difference of Hessian (Red)
    fig,ax = plt.subplots(1, 1, sharex=True, sharey=True, subplot_kw={'adjustable':'box'})
    ax.imshow(image, interpolation='nearest')
    
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    for blob_coord in blobs_doh:
        y, x, r = blob_coord
        BLOB_AREA_THRESH = (0, 5000) #Alter this for some basic analysis filtering by area
       
        if BLOB_AREA_THRESH[0] < pi*r*r and pi*r*r < BLOB_AREA_THRESH[1]:
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax.add_patch(c)
    
    plt.savefig('blobs_DoH.jpg', dpi=96*10)

    

    
    
        
def invert_image(img):
    cv2.imwrite('inverted_img.png', (255-img))
    return cv2.imread('inverted_img.png')

if __name__ == '__main__':
#    imagename = 'messi5.jpg'
#    imagename = 'bottomboard.jpg'
    imagename = 'lotsabees.png'
    imagename = 'front.jpg'
    
    #lbp_example(imagename)

    blob_example(imagename)

#    ndimg = misc.imread(imagename)
##    blobs = feature.blob_log(ndimg)
##    plot = Image.fromarray(ndimg, 'HSV') #Interesting Blob detection possible
#    img = Image.fromarray(ndimg, 'RGB')
#    gimg = img.convert('L')    
#    gimg.show() #Show Grayscale
#    
#    gimg_ex = exposure.equalize_hist(gimg)
#    feature.blob_log(gimg_ex, threshold = .3)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()