import numpy as np
import h5py
import extend as extend
from PIL import Image


#h5py data reading
f_train = h5py.File('train_catvnoncat.h5', 'r')
f_test  = h5py.File('test_catvnoncat.h5', 'r')

#Load train data
train_set_x_orig = np.array(f_train["train_set_x"][:]) # your train set features
train_set_y = np.array(f_train["train_set_y"][:]) # your train set labels

#Load test data
test_set_x_orig = np.array(f_test["test_set_x"][:]) # your train set features
test_set_y = np.array(f_test["test_set_y"][:]) # your train set labels

#Assemble data
m_train = train_set_x_orig.shape[0]
m_test  = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x = train_set_x_orig.reshape(m_train, -1).T/255
test_set_x  = test_set_x_orig.reshape(m_test, -1).T/255
classes = np.array(f_test["list_classes"][:]) # the list of classes

'''
#Extend the dataset
paths, image_add = extend.extend(num_px)
train_set_y_add = np.ones(len(paths))
train_set_x = np.block([train_set_x, image_add])
train_set_y = np.block([train_set_y, train_set_y_add])
'''