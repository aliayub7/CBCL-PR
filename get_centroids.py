# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 2020

@author: Ali Ayub
"""

import numpy as np
from copy import deepcopy
import math
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Functions_new import get_centroids
from Functions_new import check_reduce_centroids_covariances
from Functions_new import check_reduce_centroids
from sklearn.model_selection import KFold
import random
# THE FOLLOWING IS DEFINITELY NEEDED WHEN WORKING WITH PYTORCH
import os
os.environ["OMP_NUM_THREADS"] = "1"


class getCentroids:
    def __init__(self,x_train,y_train,classes,seed,centroids_limit=None,current_centroids=[],increment=None,distance_metric='euclidean',clustering_type='Agglomerative_variant',
    k_base=1,k_limit=25,x_val=None,y_val=None,get_covariances=False,complete_covariances=[],complete_centroids_num=[],d_best=17.0,diag_covariances=False):
        self.x_train = x_train
        self.y_train = y_train
        self.total_classes = classes
        self.increment = increment
        self.total_centroids_limit = centroids_limit
        self.complete_centroids = current_centroids
        self.distance_metric = distance_metric
        self.clustering_type = clustering_type
        self.k_base = k_base
        self.k_limit = k_limit
        self.best_k = None
        self.total_num = []
        self.x_val = x_val
        self.y_val = y_val
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.get_covariances = get_covariances
        self.diag_covariances = diag_covariances
        self.complete_covariances = complete_covariances
        self.complete_centroids_num = complete_centroids_num
        self.d_best = d_best

    def initialize(self,x_train,y_train,classes,seed,centroids_limit=None,current_centroids=[],increment=None,distance_metric='euclidean',clustering_type='Agglomerative_variant',
    k_base=1,k_limit=25,x_val=None,y_val=None,get_covariances=False,complete_covariances=[],complete_centroids_num=[],d_best=17.0,diag_covariances=False):
        self.x_train = x_train
        self.y_train = y_train
        self.total_classes = classes
        self.increment = increment
        self.total_centroids_limit = centroids_limit
        self.complete_centroids = current_centroids
        self.distance_metric = distance_metric
        self.clustering_type = clustering_type
        self.k_base = k_base
        self.k_limit = k_limit
        self.best_k = None
        self.x_val = x_val
        self.y_val = y_val
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.get_covariances = get_covariances
        self.diag_covariances = diag_covariances
        self.complete_covariances = complete_covariances
        self.complete_centroids_num = complete_centroids_num
        self.d_best = d_best

    def update_centroids(self):
        current_total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            current_total_centroids+=len(self.complete_centroids[i])

        x_train_temp,x_val,y_train_temp,y_val = train_test_split(self.x_train,self.y_train,test_size=0.2,stratify = self.y_train)
        train_data = [[] for y in range(self.total_classes)]
        #total_num = [0 for y in range(self.total_classes)]
        total_num = deepcopy(self.total_num)
        total_num.extend([0 for y in range(self.total_classes)])
        for i in range(0,len(y_train_temp)):
            train_data[y_train_temp[i]-(self.increment*self.total_classes)].append(x_train_temp[i])
            total_num[y_train_temp[i]]+=1
        #print ("total images per class:",total_num)

        weighting = np.divide([1 for x in range(0,(len(total_num)))],total_num)
        weighting = np.divide(weighting,np.sum(weighting))

        # get the best centroids
        self.total_num.extend([0 for y in range(self.total_classes)])
        train_data = [[] for y in range(self.total_classes)]
        for i in range(0,len(self.y_train)):
            train_data[self.y_train[i]-(self.increment*self.total_classes)].append(self.x_train[i])
            self.total_num[self.y_train[i]]+=1

        train_pack = []
        for i in range(0,self.total_classes):
            train_pack.append([train_data[i],self.d_best,self.clustering_type,self.get_covariances,self.diag_covariances])
        if self.get_covariances!=True:
            my_pool = Pool(self.total_classes)
            centroids = my_pool.map(get_centroids,train_pack)
            my_pool.close()
        else:
            my_pool = Pool(self.total_classes)
            centroids_variances = my_pool.map(get_centroids,train_pack)
            my_pool.close()
            centroids = []
            covariances = []
            centroids_num = []
            for j in range(0,len(centroids_variances)):
                centroids.append(centroids_variances[j][0])
                covariances.append(centroids_variances[j][1])
                centroids_num.append(centroids_variances[j][2])

        exp_centroids = 0
        for i in range(0,len(centroids)):
            exp_centroids+=len(centroids[i])

        # reduce previous centroids if more than allowed, THIS HAS TO BE CHANGED FOR COVARIANCES
        if self.total_centroids_limit!=None:
            self.complete_centroids,self.complete_covariances,self.complete_centroids_num = check_reduce_centroids_covariances(self.complete_centroids,self.complete_covariances,self.complete_centroids_num,
            current_total_centroids,exp_centroids,self.total_centroids_limit,self.increment,
            self.total_classes)
        # add the new centroids to the complete_centroids
        self.complete_centroids.extend(centroids)
        if self.get_covariances==True:
            self.complete_covariances.extend(covariances)
            self.complete_centroids_num.extend(centroids_num)

        print ("total_classes",len(self.complete_centroids))
        total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            total_centroids+=len(self.complete_centroids[i])
        print ("total_centroids",total_centroids)

        #self.best_k = indis[np.argmax(max_acs)]

