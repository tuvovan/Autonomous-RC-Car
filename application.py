import cv2
import numpy as np
import os
from random import  shuffle
from tqdm import tqdm


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


IM_SIZE = 150
LR = 1e-3

MODEL_NAME = 'control_car.model'

convnet = input_data(shape=[None, IM_SIZE, IM_SIZE, 1], name='input')
# http://tflearn.org/layers/conv/
# http://tflearn.org/activations/
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model load!')


img = cv2.imread('a3.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IM_SIZE, IM_SIZE))

data = img.reshape(IM_SIZE,IM_SIZE,1)

model_out = model.predict([data])[0]

if np.argmax(model_out) == 0: str_label = 'forward'
elif np.argmax(model_out) == 1: str_label = 'left'
elif np.argmax(model_out) == 2: str_label = 'right'
elif np.argmax(model_out) == 3: str_label = 'idle'

cv2.imshow('image',img)
print(str_label)