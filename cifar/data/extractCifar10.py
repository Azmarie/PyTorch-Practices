# Helper Function Adapted from: https://gist.github.com/juliensimon/273bef4c5b4490c687b2f92ee721b546
# Revise was made to for pickle, encoding, and output path by me.

import mxnet as mx
import numpy as np
import pickle
import cv2

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='latin1')
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='latin1')
    return dict['label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)

imgarray, lblarray = extractImagesAndLabels("cifar-10-batches-py/", "test_batch")
print(imgarray.shape)
print(lblarray.shape)

categories = extractCategories("cifar-10-batches-py/", "batches.meta")
print(categories)

test = []
for i in range(0,10000):
    saveCifarImage(imgarray[i], "./test/", "image"+(str)(i))
    category = lblarray[i].asnumpy()
    category = (int)(category[0])
    test.append(categories[category])
# print(test)