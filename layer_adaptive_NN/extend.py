### Training data extending ###
import numpy as np
import glob
import os
import cv2
import scipy
import scipy.misc
from scipy import ndimage

def extend(num_px):
    global paths, img_add

    image_path = '../original_image/extend'
    paths = glob.glob(os.path.join(image_path, '*.jpg'))
    paths.sort()
    img_add = np.zeros([(num_px**2)*3, len(paths)])
    
    for i, path in enumerate(paths):
        img_load = cv2.imread(path, 1)
        img_load = img_load[...,::-1]
        img_resize = cv2.resize(img_load, (num_px, num_px), interpolation = cv2.INTER_CUBIC)
        img_add[:,i] = np.squeeze(scipy.misc.imresize(img_resize, \
               size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T)/255
        
    return paths, img_add


