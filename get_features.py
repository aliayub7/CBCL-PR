import numpy as np
import sys
import os
import time
import pickle
import cv2
from img_to_vec import Img2Vec
from PIL import Image

path_to_train = './data/cifar-100-python/train'
path_to_test = './data/cifar-100-python/test'


with open(path_to_train, 'rb') as fo:
    train_batch = pickle.load(fo, encoding='latin1')

with open(path_to_test, 'rb') as fo:
    test_batch = pickle.load(fo, encoding='latin1')


train_images = train_batch['data'].reshape((len(train_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
train_labels = train_batch['fine_labels']

test_images = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
test_labels = test_batch['fine_labels']

categories = dict()
total_classes=100
total_num=[0]*total_classes

img2vec = Img2Vec(cuda=True,model='resnet-34')
train_features = []
test_features = []
iter = 0

for i in range(0,len(train_images)):
    total_num[train_labels[i]]+=1
    #img = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    vec = img2vec.get_vec(img)
    #if vec is not None:
    print ('iter',iter)
    iter+=1
    features_np=np.array(vec)
    features_f = features_np.flatten()
    train_features.append(features_f)

for i in range(0,len(test_images)):
    print ('test',i)
    #img = cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    vec = img2vec.get_vec(img)
    #if vec is not None:
    print ('iter',iter)
    iter+=1
    features_np=np.array(vec)
    features_f = features_np.flatten()
    test_features.append(features_f)

train_features = np.array(train_features)
test_features = np.array(test_features)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

with open('CIFAR_resnet34_N_train_features.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(train_features, filehandle)
with open('CIFAR_resnet34_N_test_features.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(test_features, filehandle)
with open('CIFAR_resnet34_N_train_labels.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(train_labels, filehandle)
with open('CIFAR_resnet34_N_test_labels.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(test_labels, filehandle)
