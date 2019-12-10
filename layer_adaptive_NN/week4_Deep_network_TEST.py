#week4_assignment TESTING file
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
import time
import pickle
from dataload import *
import week4_Deep_predict as w4Dp


#Load TRAINING data
parameters = pickle.load(open('well_trained_parameters_NN_cat.txt', 'rb'))

threshold = 0.85


#Analysis get started
#No.1
plt.figure()
figure_sam_1_raw = cv2.imread(('..\original_image\sam_1.jpg'), 1)
figure_sam_1_raw = figure_sam_1_raw[...,::-1]
figure_sam_1_resize = cv2.resize(figure_sam_1_raw, (num_px, num_px), interpolation = cv2.INTER_CUBIC)
plt.imshow(figure_sam_1_resize)
plt.title('It is a cat figure?', fontsize = 20)
plt.show()

figure_sam_1 = figure_sam_1_resize/255
figure_sam_1 = scipy.misc.imresize(figure_sam_1, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
tic_1 = time.clock()
##
p1, probas1, caches1 = w4Dp.predict(figure_sam_1, parameters, threshold)
##
toc_1 = time.clock()

print('################################')
print('Report for the 1st fuking figure')
print('################################')
print('There are %i hidden layers' % (len(parameters['Layer_dimension']) - 2))
print('Case 1 detection time:', (toc_1 - tic_1), 'sec')
print('Probably for being a cat:', probas1)
print("y = " + str(np.squeeze(p1)) + \
      ", your algorithm predicts a \"" + classes[int(np.squeeze(p1)),].decode("utf-8") +  \
      "\" picture.")
print('')



#No.2
plt.figure()
figure_sam_2_raw = cv2.imread(('..\original_image\sam_2.jpg'), 1)
figure_sam_2_raw = figure_sam_2_raw[...,::-1]
figure_sam_2_resize = cv2.resize(figure_sam_2_raw, (num_px, num_px), interpolation = cv2.INTER_CUBIC)
plt.imshow(figure_sam_2_resize)
plt.title('It is a cat figure?', fontsize = 20)
plt.show()

figure_sam_2 = figure_sam_2_resize/255
figure_sam_2 = scipy.misc.imresize(figure_sam_2, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
tic_2 = time.clock()
##
p2, probas2, caches2 = w4Dp.predict(figure_sam_2, parameters, threshold)
##
toc_2 = time.clock()

print('################################')
print('Report for the 2rd fuking figure')
print('################################')
print('There are %i hidden layers' % (len(parameters['Layer_dimension']) - 2))
print('Case 2 detection time:', (toc_2 - tic_2), 'sec')
print('Probably for being a cat:', probas2)
print("y = " + str(np.squeeze(p2)) + \
      ", your algorithm predicts a \"" + classes[int(np.squeeze(p2)),].decode("utf-8") +  \
      "\" picture.")
print('')



#Analysis get started
#No.1
plt.figure()
figure_sam_3_raw = cv2.imread(('..\original_image\sam_3.jpg'), 1)
figure_sam_3_raw = figure_sam_3_raw[...,::-1]
figure_sam_3_resize = cv2.resize(figure_sam_3_raw, (num_px, num_px), interpolation = cv2.INTER_CUBIC)
plt.imshow(figure_sam_3_resize)
plt.title('It is a cat figure?', fontsize = 20)
plt.show()

figure_sam_3 = figure_sam_3_resize/255
figure_sam_3 = scipy.misc.imresize(figure_sam_3, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
tic_3 = time.clock()
##
p3, probas3, caches3 = w4Dp.predict(figure_sam_3, parameters, threshold)
##
toc_3 = time.clock()

print('################################')
print('Report for the 3th fuking figure')
print('################################')
print('There are %i hidden layers' % (len(parameters['Layer_dimension']) - 2))
print('Case 1 detection time:', (toc_3 - tic_3), 'sec')
print('Probably for being a cat:', probas3)
print("y = " + str(np.squeeze(p3)) + \
      ", your algorithm predicts a \"" + classes[int(np.squeeze(p3)),].decode("utf-8") +  \
      "\" picture.")
print('')

