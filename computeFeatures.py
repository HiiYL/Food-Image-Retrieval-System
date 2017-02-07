"""
computeFeatures.py

YOUR WORKING FUNCTION for computing features

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

from cyvlfeat import sift as cysift

# you are allowed to import other Python packages above
##########################
def computeFeatures(img):
    # Inputs
    # img: 3-D numpy array of an RGB color image
    #
    # Output
    # featvect: A D-dimensional vector of the input image 'img'
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    
    # This is the baseline method: 192-D RGB colour feature histogram
    rhist, rbins = np.histogram(img[:,:,0], 64, normed=True)
    ghist, gbins = np.histogram(img[:,:,1], 64, normed=True)
    bhist, bbins = np.histogram(img[:,:,2], 64, normed=True)
    featvect = np.concatenate((rhist, ghist, bhist))

    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # f2, des2 = cysift.sift(img_gray, peak_thresh=8, edge_thresh=5, compute_descriptor=True)

    # featvect = des2

    # print(des2.shape)
    
    
    # This creates a 300-D vector of random values as features!     
    #featvect = np.random.rand(300, 1)
    
    # END OF YOUR CODE
    #########################################################################
    return featvect 