import sys
import time as t

import numpy as np
from skimage.feature import hog, BRIEF

import random as rd
from hog import hog_descriptor


def apply_hog(left, right, descriptor_size, num_orientations):
    """
    computes HOG descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: number of pixels in a hog cell.
    :param num_orientations: number of HOG orientations.
    :return: (H x W x M) array, H = height, W = width and M = num_orientations, of type np.float32.
    """
    # TODO: apply HOG descriptor on left and right images.
    return hog_descriptor(left, descriptor_size, num_orientations), hog_descriptor(right, descriptor_size, num_orientations)


def apply_brief(left, right, descriptor_size, num_elements, pad):
    """
    computes BRIEF descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: size of window of the BRIEF descriptor.
    :param num_elements: length of the feature vector.
    :return: (H x W) array, H = height and W = width, of type np.int64
    """
    
    # TODO: apply BRIEF descriptor on both images. You will have to convert the BRIEF feature vector to a int64.
    imSize = left.shape
    height = imSize[0]
    width = imSize[1]
    
    keyPoints = np.array([[i % height, int(i/height)] for i in range(height*width)], np.int32)
    
    leftExtractor = BRIEF(descriptor_size = num_elements, patch_size= descriptor_size*descriptor_size, mode='normal')
    rightExtractor = BRIEF(descriptor_size = num_elements, patch_size= descriptor_size*descriptor_size, mode='normal')
    
    leftExtractor.extract(left, keyPoints)   
    rightExtractor.extract(right, keyPoints)
    
    leftKeyPoints = keyPoints[leftExtractor.mask]
    rightKeyPoints = keyPoints[rightExtractor.mask]
    
    leftOutput = np.zeros((height, width), np.int64)
    rightOutput = np.zeros((height, width), np.int64)
    
    leftDescCount = 0
    rightDescCount = 0
    
    for point in leftKeyPoints:
        i = point[0]
        j = point[1]
        value = 0
        for k in range(num_elements):
            if(leftExtractor.descriptors[leftDescCount][num_elements - 1 - k]):
                value += pow(2, k)
        leftOutput[i][j] = np.int64(value)
        leftDescCount += 1
        
    for point in rightKeyPoints:
        i = point[0]
        j = point[1]
        value = 0
        for k in range(num_elements):
            if(rightExtractor.descriptors[rightDescCount][num_elements - 1 - k]):
                value += pow(2, k)
        rightOutput[i][j] = np.int64(value)
        rightDescCount += 1
        
#    for i in range(height):
#        for j in range(width - 1):
#            leftVal = leftOutput[i][j]
#            leftNeighbour = leftOutput[i][j+1]
#            if(leftVal == 0 and leftNeighbour != 0):
#                leftOutput[i][j] = leftNeighbour
#                
#            rightVal = rightOutput[i][j]
#            rightNeighbour = rightOutput[i][j+1]
#            if(rightVal == 0 and rightNeighbour != 0):
#                rightOutput[i][j] = rightNeighbour
            
    
#    
#    for i in range(height):
#        for j in range(width):
#            id = j*height + i
#            if(leftExtractor.mask[id]):
#                value = 0
#                for k in range(num_elements):
#                    if(leftExtractor.descriptors[leftDescCount][num_elements - 1 - k]):
#                        value += pow(2, k)
#                leftOutput[i][j] = np.int64(value)
#                leftDescCount += 1
#            else:
#                leftOutput[i][j] = np.int64(rd.randint(0, pow(2, num_elements) - 1))
#            if(rightExtractor.mask[id]):
#                value = 0
#                for k in range(num_elements):
#                    if(rightExtractor.descriptors[rightDescCount][num_elements - 1 - k]):
#                        value += pow(2, k)
#                rightOutput[i][j] = np.int64(value)
#                rightDescCount += 1
#            else:
#                rightOutput[i][j] = np.int64(rd.randint(0, pow(2, num_elements) - 1))
#            
    print(leftDescCount, rightDescCount)

    leftDesc = leftOutput[pad:(height - pad), pad:(width - pad)]
    rightDesc = rightOutput[pad:(height - pad), pad:(width - pad)]
    
    print(height, width)
    print(leftOutput.shape, rightOutput.shape)
    print(leftDesc.shape, rightDesc.shape)
            
    return leftDesc, rightDesc