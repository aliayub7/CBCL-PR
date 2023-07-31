# -*- coding: utf-8 -*-
"""
Created on Wed Jul 3 2019

@author: Ali Ayub
"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
#import math
#from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Functions_new import get_test_accuracy
#from sklearn.decomposition import PCA
#from sklearn.model_selection import KFold
import pandas as pd

from get_centroids import getCentroids
from get_incremental_data import getIncrementalData

import torch
from torch.autograd import Variable
import torch.nn as nn
#from torch.optim import SGD
import torch.optim as optim
from my_model import Net
from training_functions import train_model
from training_functions import eval_model

import json
import random
import time


def main(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # load features and labels
    with open('./features/CIFAR_Resnet34_train_features.data', 'rb') as filehandle:
        # read the data as binary data stream
        x_train = pickle.load(filehandle)
    with open('./features/CIFAR_Resnet34_test_features.data', 'rb') as filehandle:
        # read the data as binary data stream
        x_test = pickle.load(filehandle)
    with open('./features/CIFAR_Resnet34_train_labels.data', 'rb') as filehandle:
        # read the data as binary data stream
        y_train = pickle.load(filehandle)
    with open('./features/CIFAR_Resnet34_test_labels.data', 'rb') as filehandle:
        # read the data as binary data stream
        y_test = pickle.load(filehandle)

    distance_metric = 'euclidean'
    clustering_type = 'Agglomerative_variant'
    full_classes = 100
    total_classes = 2
    k_base = 1
    k_limit = 1
    d_best = 17.0
    total_centroids_limit = None
    get_covariances = True
    diag_covariances = True
    
    # for few-shot
    k_shot = None
    # how many pseudo-samples per class
    samples_per = 40

    # pytorch stuff
    # deifine model
    model = Net(dim=len(x_train[0]),total_classes=total_classes,seed=seed)

    num_epochs = 25
    batch_size = 64
    criterion = nn.CrossEntropyLoss()

    # no learning rate schedule right now
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print (device)

    save_data = True
    features_name = "Resnet34_fullset_"+str(total_classes)+"classes_1layer_diag"

    complete_x_test = []
    complete_y_test = []
    complete_centroids = []
    complete_covariances = []
    complete_centroids_num = []

    # get train, test splits
    #x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,stratify=labels)
    # get incremental data
    incremental_data_creator = getIncrementalData(x_train,y_train,x_test,y_test,full_classes=full_classes,seed=seed)
    incremental_data_creator.incremental_data(total_classes=total_classes,limiter=full_classes)
    train_features_increment = incremental_data_creator.train_features_increment
    train_labels_increment = incremental_data_creator.train_labels_increment
    test_features_increment = incremental_data_creator.test_features_increment
    test_labels_increment = incremental_data_creator.test_labels_increment

    base_test_features = test_features_increment[0]
    base_test_labels = test_labels_increment[0]

    experimental_data = dict()

    centroid_finder = getCentroids(train_features_increment[0],train_labels_increment[0],total_classes,seed=seed,get_covariances=get_covariances)

    for increment in range(0,int(full_classes/total_classes)):
        print (' ')

        print ('number of training images',len(train_labels_increment[increment]))
        
        if k_shot is not None:
            # For few-shot
            x_train_2,x_test2,y_train_2,y_test2 = train_test_split(train_features_increment[increment],train_labels_increment[increment],test_size=0.2,
            stratify=train_labels_increment[increment])
            total_num_temp = [0 for x in range(0,total_classes)]
            x_train_increment = []
            y_train_increment = []
            for i in range(0,len(y_train_2)):
                if total_num_temp[y_train_2[i]-(increment*total_classes)]<k_shot:
                    total_num_temp[y_train_2[i]-(increment*total_classes)]+=1
                    x_train_increment.append(x_train_2[i])
                    y_train_increment.append(y_train_2[i])
            print ('number of training images',len(y_train_increment))
            print ('number of test images',len(test_labels_increment[increment]))
        
        if k_shot is None:
            # initialize with new incremental data
            centroid_finder.initialize(train_features_increment[increment],train_labels_increment[increment],classes = total_classes,seed = seed,
            current_centroids=complete_centroids,increment = increment,get_covariances=get_covariances,complete_covariances=complete_covariances,
            complete_centroids_num=complete_centroids_num,d_best=d_best,diag_covariances=diag_covariances,clustering_type=clustering_type,centroids_limit=total_centroids_limit)
        else:
            # for few shot
            centroid_finder.initialize(x_train_increment,y_train_increment,classes = total_classes,seed = seed,
            current_centroids=complete_centroids,increment = increment, complete_covariances = complete_covariances,get_covariances=get_covariances,
            complete_centroids_num=complete_centroids_num)

        # find the centroids for new classes
        centroid_finder.validation_based()

        # updating variables from the centroid_finder class
        complete_centroids = centroid_finder.complete_centroids
        complete_covariances = centroid_finder.complete_covariances
        complete_centroids_num = centroid_finder.complete_centroids_num
        k = centroid_finder.best_k
        total_num = centroid_finder.total_num

        weighting = np.divide([1 for x in range(0,(len(total_num)))],total_num)
        weighting = np.divide(weighting,np.sum(weighting))

        # create the full test set
        complete_x_test.extend(test_features_increment[increment])
        complete_y_test.extend(test_labels_increment[increment])

        # Now the Pytorch phase
        if increment == 0:
            # convert training set into torch format
            x_train = np.array(train_features_increment[increment])
            y_train = np.array(train_labels_increment[increment])
            x_train = x_train.reshape(len(train_features_increment[increment]),len(train_features_increment[increment][0]))
            x_train = torch.from_numpy(x_train)
            x_train = x_train.float()
            y_train = y_train.astype(int)
            y_train = torch.from_numpy(y_train)

        else:
            # generate data from the previous centroids
            previous_centroids = complete_centroids[0:total_classes+((increment-1)*total_classes)]
            previous_centroids_num = complete_centroids_num[0:total_classes+((increment-1)*total_classes)]
            previous_covariances = complete_covariances[0:total_classes+((increment-1)*total_classes)]
            samples_per_class = total_num[0:total_classes+((increment-1)*total_classes)]
            previous_samples = []
            previous_labels = []
            for i in range(0,len(samples_per_class)):
                if k_shot is not None:
                    ones_count = complete_centroids_num[i].count(1)
                    req_samples = samples_per - ones_count
                    if len(complete_centroids_num[i])-ones_count>0:
                        how_many_per_centroid = round(req_samples/(len(complete_centroids_num[i])-ones_count))
                else:
                    how_many_per_centroid = previous_centroids_num[i][j]
                for j in range(0,len(previous_centroids_num[i])):
                    if previous_centroids_num[i][j]>1:
                        if diag_covariances != True:
                            temp = list(np.random.multivariate_normal(previous_centroids[i][j],previous_covariances[i][j],how_many_per_centroid))
                        else:
                            #if increment >0:
                            #    print ('cent shape',previous_centroids[i][j].shape)
                            #    print ('cov shape',previous_covariances[i][j].shape)
                            temp = list(np.random.multivariate_normal(previous_centroids[i][j],np.diag(previous_covariances[i][j]),how_many_per_centroid))
                        previous_samples.extend(temp)
                        previous_labels.extend([i for x in range(0,how_many_per_centroid)])
                    else:
                        previous_samples.append(previous_centroids[i][j])
                        previous_labels.append(i)
            # convert training set into torch format
            x_train = previous_samples
            y_train = previous_labels
            x_train.extend(train_features_increment[increment])
            y_train.extend(train_labels_increment[increment])
            number_of_samples = len(x_train)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_train = x_train.reshape(number_of_samples,len(train_features_increment[increment][0]))
            x_train = torch.from_numpy(x_train)
            x_train = x_train.float()
            y_train = y_train.astype(int)
            y_train = torch.from_numpy(y_train)

        # covert test set into torch format
        x_test = np.array(complete_x_test)
        y_test = np.array(complete_y_test)
        x_test = x_test.reshape(len(complete_x_test),len(complete_x_test[0]))
        x_test = torch.from_numpy(x_test)
        x_test = x_test.float()
        y_test = y_test.astype(int)
        y_test = torch.from_numpy(y_test)

        model = Net(dim=len(train_features_increment[increment][0]),total_classes=total_classes+(total_classes*increment),seed=seed)
        #model.fc_layers[3] = nn.Linear(len(train_features_increment[increment][0]),total_classes+(total_classes*increment))
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        model = model.to(device)

        since = time.time()
        model,best_acc = train_model(x_train,y_train,x_test,y_test,model,criterion,optimizer,device,seed,num_epochs=num_epochs,batch_size = batch_size)
        best_acc = (best_acc.cpu().numpy().tolist())
        print ('training time',time.time()-since)

        # Returning the memory
        train_features_increment[increment] = None
        train_labels_increment[increment] = None
        test_features_increment[increment] = None
        test_labels_increment[increment] = None
        incremental_data_creator = None
        previous_centroids = None
        previous_covariances = None
        previous_centroids_num = None
        previous_samples = None
        previous_labels = None
        samples_per_class = None
        number_of_samples = None

        experimental_data[increment] = dict()
        experimental_data[increment]['total_classes'] = len(complete_centroids)
        experimental_data[increment]['model_test_accuracy'] = best_acc
        experimental_data[increment]['seed'] = seed

    if save_data == True:
        with open('data.json','r') as f:
            data=json.load(f)
        if features_name not in data:
            data[features_name] = dict()
        data[features_name][str(len(data[features_name])+1)] = experimental_data
        with open('data.json', 'w') as fp:
            json.dump(data, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
    seed = random.randint(1, 1000)
    main(seed)
