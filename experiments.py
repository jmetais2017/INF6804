import cv2
import numpy as np
import math

import sgm
import distances


def get_stats(disparity, gt, max_disp, pad, descriptor):
    """
    computes the recall of the disparity map.
    :param disparity: disparity image.
    :param gt: path to ground-truth image.
    :param max_disp: maximum disparity.
    :return: rate of correct predictions.
    """
    gt = np.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
    
    if descriptor == 'brief':
        imSize = gt.shape
        height = imSize[0]
        width = imSize[1]
        gt = gt[pad:(height - pad), pad:(width - pad)]
    
    gt = np.int16(gt / 255.0 * float(max_disp))
    disparity = np.int16(np.float32(disparity) / 255.0 * float(max_disp))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    mean = np.mean(np.abs(disparity - gt))
    sigma = math.sqrt(np.var(np.abs(disparity - gt)))
    return float(correct) / gt.size, mean, sigma


def experiments():
    left = 'cones/im2.png'
    right = 'cones/im6.png'
    gt_name = 'cones/disp2.png'
    
    descriptor = 'brief'
    if descriptor == 'brief':
        distance = distances.hamming_distance
    else:
        distance = distances.l2_distance
    descriptor_size = 8
    num_elements = 63
    num_orientation = 8
    max_disp = 64
    pad = 40 

    disparity_map = sgm.sgm(left, right, descriptor, distance, descriptor_size, num_elements, num_orientation, max_disp, pad)
    cv2.imwrite('left_disp_map.png', disparity_map) 
    
    print('\nEvaluating left disparity map...')
    recall, mean, sigma = get_stats(disparity_map, gt_name, max_disp, pad, descriptor)
    print('\tRecall = {:.2f}%'.format(recall * 100.0))
    print('\tMean difference = {:.2f}'.format(mean))
    print('\tEcart-type = {:.2f}'.format(sigma))


#if __name__ == '__main__':
experiments()
