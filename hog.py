import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray

def hog_descriptor(image, descriptor_size, num_orientations):
    
    image = rgb2gray(image)
    h = len(image)
    w = len(image[0])

    #pixels per cell
    ppc = (descriptor_size, descriptor_size)
    
    fd = np.zeros((h, w, num_orientations), np.float32)
    
    #Construction des images decalees
    image2 = np.zeros((h+1, w))
    image3 = np.zeros((h, w+1)) 
    image4 = np.zeros((h+1, w+1))          
    for i in range(h):
        for j in range(w): 
            image2[i+1][j] = image[i][j]
            image3[i][j+1] = image[i][j]
            image4[i+1][j+1] = image[i][j]
    image2 = image2[:h]
    image3 = image3[:][:w]
    image4 = image4[:h][:w]
    
    #HOG dans les 4 images
    fd1 = hog(image, orientations=num_orientations, pixels_per_cell=ppc,
              cells_per_block=(1, 1), visualize=False,     
              feature_vector=False, multichannel=False)
    fd2 = hog(image2, orientations=num_orientations, pixels_per_cell=ppc,
              cells_per_block=(1, 1), visualize=False,
              feature_vector=False, multichannel=False)
    fd3 = hog(image3, orientations=num_orientations, pixels_per_cell=ppc,
              cells_per_block=(1, 1), visualize=False,
              feature_vector=False, multichannel=False)
    fd4 = hog(image4, orientations=num_orientations, pixels_per_cell=ppc,
              cells_per_block=(1, 1), visualize=False,
              feature_vector=False, multichannel=False)
    
    for i in range(1,len(fd1)-1):
        for j in range(1,len(fd1[0])-1):
            fd[2*i][2*j]=fd1[i][j]
            fd[2*i+1][2*j]=fd2[i][j]
            fd[2*i][2*j+1]=fd3[i][j]
            fd[2*i+1][2*j+1]=fd4[i][j]
    
    return fd