"""
Created on 03/11/2020

@author: Ali Ayub
"""
# THIS FILE IS ONLY FOR INCREMENTAL LEARNING

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from copy import deepcopy
import math
from multiprocessing import Pool
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fcluster, ward, average, weighted, complete, single
from scipy.spatial.distance import pdist

distance_metric = 'euclidean'
def get_centroids(train_pack):
    # unpack x_train
    x_train = train_pack[0]
    distance_threshold = train_pack[1]
    clustering_type = train_pack[2]
    get_covariances = train_pack[3]
    diag_covariances = train_pack[4]

    if clustering_type == 'Agglomerative':
        dist_mat=pdist(x_train,metric='euclidean')
        Z = weighted(dist_mat)
        dn = hierarchy.dendrogram(Z)
        labels=fcluster(Z, t=distance_threshold, criterion='distance')
        centroids = [[] for y in range(0,max(labels))]
        total_num = [0 for x in range(0,max(labels))]
        per_labels = [[] for y in range(0,max(labels))]
        for j in range(len(x_train)):
            per_labels[labels[j]-1].append(x_train[j])
            total_num[labels[j]-1]+=1
        covariances = [[] for y in range(0,max(labels))]

        for j in range(0,max(labels)):
            centroids[j] = np.mean(per_labels[j],0)
            if get_covariances==True:
                if diag_covariances != True:
                    covariances[j] = np.cov(np.array(per_labels[j]).T)
                else:
                    temp = np.cov(np.array(per_labels[j]).T)
                    covariances[j] = temp.diagonal()
        #for j in range(0,len(x_train)):
        #    centroids[labels[j]-1]+=x_train[j]
        #    total_number[labels[j]-1]+=1
        #for j in range(0,len(centroids)):
        #    centroids[j] = np.divide(centroids[j],total_number[j])

    elif clustering_type == 'Agglomerative_variant':
        # for each training sample do the same stuff...
        if len(x_train)>0:
            centroids = [[0 for x in range(len(x_train[0]))]]
            for_cov = [[]]

            # initalize centroids
            centroids[0] = x_train[0]
            for_cov[0].append(x_train[0])
            total_num = [1]
            for i in range(1,len(x_train)):
                distances=[]
                indices = []
                for j in range(0,len(centroids)):
                    d = find_distance(x_train[i],centroids[j],distance_metric)
                    if d<distance_threshold:
                        distances.append(d)
                        indices.append(j)
                if len(distances)==0:
                    centroids.append(x_train[i])
                    total_num.append(1)
                    for_cov.append([])
                    for_cov[len(for_cov)-1].append(list(x_train[i]))
                else:
                    #min_d = np.argmin(distances)
                    #centroids[indices[min_d]] = np.add(centroids[indices[min_d]],x_train[i])
                    #total_num[indices[min_d]]+=1
                    min_d = np.argmin(distances)
                    centroids[indices[min_d]] = np.add(np.multiply(total_num[indices[min_d]],centroids[indices[min_d]]),x_train[i])
                    total_num[indices[min_d]]+=1
                    centroids[indices[min_d]] = np.divide(centroids[indices[min_d]],(total_num[indices[min_d]]))
                    for_cov[indices[min_d]].append(list(x_train[i]))
            # calculate covariances
            if get_covariances==True:
                covariances = deepcopy(for_cov)
                for j in range(0,len(for_cov)):
                    if total_num[j]>1:
                        if diag_covariances != True:
                            covariances[j] = np.cov(np.array(for_cov[j]).T)
                        else:
                            temp = np.cov(np.array(for_cov[j]).T)
                            covariances[j] = temp.diagonal()
                    else:
                        covariances[j] = np.array([1.0 for x in range(0,len(x_train[0]))])

                #or j in range(0,len(total_num)):
                #    centroids[j]=np.divide(centroids[j],total_num[j])
        else:
            centroids = []

    elif clustering_type == 'k_means':
        kmeans = KMeans(n_clusters=distance_threshold, random_state = 0).fit(x_train)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        total_num = [0 for x in range(0,max(labels)+1)]
        per_labels = [[] for y in range(0,max(labels)+1)]
        for j in range(len(x_train)):
            per_labels[labels[j]].append(x_train[j])
            total_num[labels[j]]+=1
        covariances = [[] for y in range(0,max(labels)+1)]
        for j in range(0,max(labels)+1):
            if get_covariances==True:
                if diag_covariances != True:
                    covariances[j] = np.cov(np.array(per_labels[j]).T)
                else:
                    temp = np.cov(np.array(per_labels[j]).T)
                    covariances[j] = temp.diagonal()
    elif clustering_type == 'NCM':
        centroids = [[0 for x in range(len(x_train[0]))]]
        centroids[0] = np.average(x_train,0)

    if get_covariances == True:
        return [centroids,covariances,total_num]
    else:
        return centroids

def find_distance(data_vec,centroid,distance_metric):
    if distance_metric=='euclidean':
        return np.linalg.norm(data_vec-centroid)
    elif distance_metric == 'euclidean_squared':
        return np.square(np.linalg.norm(data_vec-centroid))
    elif distance_metric == 'cosine':
        return distance.cosine(data_vec,centroid)

# reduce give centroids using k-means
def reduce_centroids(centroid_pack):
    centroids = centroid_pack[0]
    reduction_per_class = centroid_pack[1]
    n_clusters = len(centroids) - reduction_per_class
    if n_clusters>0:

        #out_centroids = get_centroids([centroids,n_clusters,'k_means',True,False])
        kmeans = KMeans(n_clusters=n_clusters, random_state = 0).fit(centroids)
        out_centroids = kmeans.cluster_centers_

        # simple reduction
        #out_centroids = deepcopy(centroids)
        #del out_centroids[0:reduction_per_class]
    else:
        out_centroids = centroids
    return out_centroids

# check if the centroids should be reduced and reduce them
def check_reduce_centroids(temp_complete_centroids,current_total_centroids,temp_exp_centroids,total_centroids_limit,increment,total_classes):

    if current_total_centroids + temp_exp_centroids > total_centroids_limit:
        reduction_centroids = current_total_centroids + temp_exp_centroids - total_centroids_limit
        classes_so_far = increment*total_classes
        centroid_pack = []
        for i in range(0,len(temp_complete_centroids)):
            reduction_per_class = round((len(temp_complete_centroids[i])/current_total_centroids)*reduction_centroids)
            centroid_pack.append([temp_complete_centroids[i],reduction_per_class])
        my_pool = Pool(len(temp_complete_centroids))
        temp_complete_centroids = my_pool.map(reduce_centroids,centroid_pack)
        my_pool.close()
    return temp_complete_centroids

# reduce given centroids and covariances using k-means. CURRENTLY ONLY FOR DIAGONAL COVARIANCES
def reduce_centroids_covariances(centroid_pack):
    centroids = centroid_pack[0]
    covariances = centroid_pack[1]
    centroid_num = centroid_pack[2]
    reduction_per_class = centroid_pack[3]
    n_clusters = len(centroids) - reduction_per_class

    if n_clusters>0:

        #out_centroids = get_centroids([centroids,n_clusters,'k_means',False])
        kmeans = KMeans(n_clusters=n_clusters, random_state = 0).fit(centroids)
        out_centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        out_covariances = []
        covariances = np.array(covariances)
        centroid_num = np.array(centroid_num)
        out_centroids_num = []
        #print ('these are the labels',labels)
        for i in range(0,n_clusters):
            indices = [x for x,j in enumerate(labels) if j==i]
            temp = covariances[indices]
            temp_num = centroid_num[indices]
            #print ('before tem shape',temp.shape)
            #print ('this is temp',temp)
            #print ('this is temp_num',temp_num)
            #print ('this is covariances',covariances)
            #input('continue')
            if 1 in temp_num:
                if(len(set(temp_num))==1):
                    #print (temp_num)
                    out_covariances.append(temp[0])
                else:
                    temp = np.delete(temp,np.argwhere(temp_num==1),0)
                    #print ('temp',temp.shape)
                    out_covariances.append(np.mean(temp,0))
                    #print ('this is cov shape',np.array(out_covariances).shape)
                    #print ('this is cov',out_covariances)
                    #input ('continue')
            else:
                out_covariances.append(np.mean(temp,0))
            out_centroids_num.append(np.sum(temp_num,0))

        #out_centroids = deepcopy(centroids)
        #out_covariances = deepcopy(covariances)
        #out_centroids_num = deepcopy(centroid_num)
        #del out_centroids[0:reduction_per_class]
        #del out_covariances[0:reduction_per_class]
        #del out_centroids_num[0:reduction_per_class]
    else:
        out_centroids = centroids
        out_covariances = covariances
        out_centroids_num = centroid_num

    # simple reduction

    return out_centroids,out_covariances,out_centroids_num

# check if the centroids should be reduced and reduce them
def check_reduce_centroids_covariances(temp_complete_centroids,temp_complete_covariances,temp_complete_centroid_num,
    current_total_centroids,temp_exp_centroids,total_centroids_limit,increment,total_classes):
    if current_total_centroids + temp_exp_centroids > total_centroids_limit:
        reduction_centroids = current_total_centroids + temp_exp_centroids - total_centroids_limit
        classes_so_far = increment*total_classes
        centroid_pack = []
        for i in range(0,len(temp_complete_centroids)):
            reduction_per_class = round((len(temp_complete_centroids[i])/current_total_centroids)*reduction_centroids)
            centroid_pack.append([temp_complete_centroids[i],temp_complete_covariances[i],temp_complete_centroid_num[i],reduction_per_class])
        my_pool = Pool(len(temp_complete_centroids))
        outer = my_pool.map(reduce_centroids_covariances,centroid_pack)
        my_pool.close()

        #outer = []
        #for i in range(0,len(centroid_pack)):
        #    outer.append(reduce_centroids_covariances(centroid_pack[i]))
        #input ('continue')
        temp_complete_centroids = []
        temp_complete_covariances = []
        temp_complete_centroid_num = []
        for i in range(0,len(outer)):
            temp_complete_centroids.append(outer[i][0])
            temp_complete_covariances.append(outer[i][1])
            temp_complete_centroid_num.append(outer[i][2])
    return temp_complete_centroids,temp_complete_covariances,temp_complete_centroid_num


def predict_multiple_class(data_vec,centroids,class_centroid,distance_metric):
    dist = [[0,class_centroid] for x in range(len(centroids))]
    for i in range(0,len(centroids)):
        dist[i][0] = find_distance(data_vec,centroids[i],distance_metric)
    return dist

def predict_multiple(data_vec,centroids,distance_metric,tops,weighting):
    dist = []
    for i in range(0,len(centroids)):
        temp = predict_multiple_class(data_vec,centroids[i],i,distance_metric)
        dist.extend(temp)
    sorted_dist = sorted(dist)
    common_classes = [0]*len(centroids)
    if tops>len(sorted_dist):
        tops = len(sorted_dist)
    for i in range(0,tops):
        if sorted_dist[i][0]==0.0:
            common_classes[sorted_dist[i][1]] += 1
        else:
            common_classes[sorted_dist[i][1]] += ((1/(i+1))*
                                                ((sorted_dist[len(sorted_dist)-1][0]-sorted_dist[i][0])/(sorted_dist[len(sorted_dist)-1][0]-sorted_dist[0][0])))
    common_classes = np.divide(common_classes,sum(common_classes))
    common_classes = np.multiply(common_classes,weighting)
    return np.argmax(common_classes)

def get_test_accuracy(test_pack):
    x_test = test_pack[0]
    y_test = test_pack[1]
    centroids = test_pack[2]
    k = test_pack[3]
    total_classes = test_pack[4]
    weighting = test_pack[5]
    t_acc = []
    predicted_label = -1
    accus = [0]*total_classes
    total_labels = [0]*total_classes
    acc=0
    for i in range(0,len(y_test)):
        total_labels[y_test[i]]+=1
        predicted_label=predict_multiple(x_test[i],centroids,'euclidean',k,weighting)
        if predicted_label == y_test[i]:
            accus[y_test[i]]+=1
    for i in range(0,total_classes):
        accus[i] = accus[i]/total_labels[i]
    acc = np.mean(accus)
    return acc

