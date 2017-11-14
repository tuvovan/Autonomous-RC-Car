import cv2
import numpy as np
import os
from random import  shuffle
from tqdm import tqdm


TRAIN_DIR = 'E:\STUDY\Code Pycharm\classify/image'
TEST_DIR = 'E:\STUDY\Code Pycharm\classify/test'
IM_SIZE = 150
H = 150
W = 150
LR = 1e-3

MODEL_NAME = 'control_car.model'

#define label

def label_img(img):
    label = img[:2]
    if label == 'fw' : return [1,0,0,0]
    elif label == 'id' : return  [0,0,0,1]
    elif label == 'tl' : return  [0,1,0,0]
    elif label == 'tr' : return  [0,0,1,0]

#create train data
def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IM_SIZE,IM_SIZE))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy',train_data)
    return train_data


def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img[4]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IM_SIZE, IM_SIZE))
        test_data.append([np.array(img), img_num])

    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data

train_data = create_train_data()

#train model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Building convolutional convnet
convnet = input_data(shape=[None, H, W, 1], name='input')
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

train = train_data[:-1000]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,H,W,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,H,W,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id='MODEL_NAME')

model.save(MODEL_NAME)